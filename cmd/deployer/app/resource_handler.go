// Copyright 2022 Cisco Systems, Inc. and its affiliates
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

package app

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/cenkalti/backoff/v4"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"

	"github.com/cisco-open/flame/pkg/openapi"
	pbNotify "github.com/cisco-open/flame/pkg/proto/notification"
	"github.com/cisco-open/flame/pkg/restapi"
)

type resourceHandler struct {
	apiserverEp string
	notifierEp  string
	spec        openapi.ComputeSpec

	stream pbNotify.DeployEventRoute_GetDeployEventClient

	grpcDialOpt grpc.DialOption
}

func NewResourceHandler(apiserverEp string, notifierEp string, computeSpec openapi.ComputeSpec,
	bInsecure bool, bPlain bool) *resourceHandler {
	var grpcDialOpt grpc.DialOption

	if bPlain {
		grpcDialOpt = grpc.WithTransportCredentials(insecure.NewCredentials())
	} else {
		tlsCfg := &tls.Config{}
		if bInsecure {
			zap.S().Warn("Warning: allow insecure connection\n")

			tlsCfg.InsecureSkipVerify = true
			http.DefaultTransport.(*http.Transport).TLSClientConfig = tlsCfg
		}
		grpcDialOpt = grpc.WithTransportCredentials(credentials.NewTLS(tlsCfg))
	}

	rHandler := &resourceHandler{
		apiserverEp: apiserverEp,
		notifierEp:  notifierEp,
		spec:        computeSpec,
		grpcDialOpt: grpcDialOpt,
	}

	return rHandler
}

// start connects to the notifier via grpc and handles notifications from the notifier
func (r *resourceHandler) Start() {
	go r.doStart()
}

func (r *resourceHandler) doStart() {
	pauseTime := 10 * time.Second

	for {
		expBackoff := backoff.NewExponentialBackOff()
		expBackoff.MaxElapsedTime = 5 * time.Minute // max wait time: 5 minutes
		err := backoff.Retry(r.connect, expBackoff)
		if err != nil {
			zap.S().Fatalf("Cannot connect with notifier: %v", err)
		}

		r.do()

		// if connection is broken right after connection is made, this can cause
		// too many connection/disconnection events. To migitage that, add some static
		// pause time.
		time.Sleep(pauseTime)
	}
}

func (r *resourceHandler) connect() error {
	// dial server
	conn, err := grpc.Dial(r.notifierEp, r.grpcDialOpt)
	if err != nil {
		zap.S().Debugf("Cannot connect with notifier: %v, conn: %v", err, conn)
		return err
	}
	zap.S().Infof("Connected with notifier at: %s, will open stream", r.notifierEp)

	client := pbNotify.NewDeployEventRouteClient(conn)
	in := &pbNotify.DeployInfo{
		ComputeId: r.spec.ComputeId,
		ApiKey:    r.spec.ApiKey,
	}

	// setup notification stream
	stream, err := client.GetDeployEvent(context.Background(), in)
	if err != nil {
		zap.S().Debugf("Open stream error: %v", err)
		return err
	}

	r.stream = stream
	zap.S().Infof("Opened stream with notifier at %s", r.notifierEp)

	return nil
}

func (r *resourceHandler) do() {
	for {
		resp, err := r.stream.Recv()
		if err != nil {
			zap.S().Errorf("Failed to receive notification: %v", err)
			break
		}

		r.dealWith(resp)
	}

	zap.S().Info("Disconnected from notifier")
}

func (r *resourceHandler) dealWith(in *pbNotify.DeployEvent) {
	switch in.GetType() {
	case pbNotify.DeployEventType_ADD_RESOURCE:
		r.addResource(in.JobId)

	case pbNotify.DeployEventType_REVOKE_RESOURCE:
		r.revokeResource(in.JobId)

	case pbNotify.DeployEventType_UNKNOWN_DEPLOYMENT_TYPE:
		fallthrough
	default:
		zap.S().Errorf("Invalid message type: %s", in.GetType())
	}
}

func (r *resourceHandler) addResource(jobId string) {
	zap.S().Infof("Received add resource request for job %s", jobId)

	// Sending request to apiserver to get deployment config for specific jobId and computeId
	deploymentConfig, err := r.getDeploymentConfig(jobId)
	if err != nil {
		fmt.Printf("Failed to get deploymentConfig for job %s: %v\n", jobId, err)
	}
	zap.S().Infof("Got deployment config from apiserver: %v", deploymentConfig)

	// Deploy resources (agents) for the job based on the configuration
	err = r.deployResources(deploymentConfig)
	if err != nil {
		fmt.Printf("Failed to deploy resources for job %s: %v\n", jobId, err)
	}
	zap.S().Infof("Successfully added resources for compute %s and jobId %s", r.spec.ComputeId, jobId)
}

func (r *resourceHandler) revokeResource(jobId string) {
	zap.S().Infof("Received revoke resource request for job %s", jobId)
	// TODO: implement revokeResource method
}

func (r *resourceHandler) getDeploymentConfig(jobId string) (openapi.DeploymentConfig, error) {
	zap.S().Infof("Sending request to apiserver / controller to get deployment config")
	// construct url
	uriMap := map[string]string{
		"computeId": r.spec.ComputeId,
		"jobId":     jobId,
	}
	url := restapi.CreateURL(r.apiserverEp, restapi.GetDeploymentConfigEndpoint, uriMap)
	code, respBody, err := restapi.HTTPGet(url)
	if err != nil || restapi.CheckStatusCode(code) != nil {
		fmt.Printf("Deployer failed to get deployment config for job %s - code: %d, error: %v\n", jobId, code, err)
		return openapi.DeploymentConfig{}, nil
	}

	deploymentConfig := openapi.DeploymentConfig{}
	err = json.Unmarshal(respBody, &deploymentConfig)
	if err != nil {
		fmt.Printf("WARNING: Failed to parse resp message: %v", err)
		return openapi.DeploymentConfig{}, nil
	}
	return deploymentConfig, nil
}

func (r *resourceHandler) deployResources(deploymentConfig openapi.DeploymentConfig) error {
	zap.S().Infof("Beginning deployment of agents")

	// iterate over agent kvs and invoke install for each
	// Create more functions and move code from /cmd/controller/deployer/k8s.go

	zap.S().Infof("Completed deployment of agents")
	return nil
}
