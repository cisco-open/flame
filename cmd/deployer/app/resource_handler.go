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

func (r *resourceHandler) revokeResource(jobId string) error {
	zap.S().Infof("Received revoke resource request for job %s", jobId)

	// WORK IN PROGRESS: Moved code directly from controller. Contains only core logic which could be used.
	// However, it needs to be re-written to be used in the standalone deployer.

	// // 1. decommission compute resources if they are in use
	// if h.dplyr != nil {
	// 	if err := h.dplyr.Uninstall("job-" + h.jobId); err != nil {
	// 		zap.S().Warnf("failed to release resources for job %s: %v", h.jobId, err)
	// 	}
	// }

	// // 2.delete all the job resource specification files
	// deploymentChartPath := filepath.Join(deploymentDirPath, h.jobId)
	// _ = os.RemoveAll(deploymentChartPath)

	// h.tskWatchCancelFn()

	zap.S().Infof("Completed revocation of agent resources")
	return nil
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

	// WORK IN PROGRESS: Moved code directly from controller. Contains only core logic which could be used.
	// However, it needs to be re-written to be used in the standalone deployer.

	// deploymentChartPath := filepath.Join(deploymentDirPath, h.jobId)
	// targetTemplateDirPath := filepath.Join(deploymentChartPath, deploymentTemplateDir)
	// if err := os.MkdirAll(targetTemplateDirPath, util.FilePerm0644); err != nil {
	// 	errMsg := fmt.Sprintf("failed to create a deployment template folder: %v", err)
	// 	zap.S().Debugf(errMsg)

	// 	return fmt.Errorf(errMsg)
	// }

	// // Copy helm chart files to destination folder
	// for _, chartFile := range helmChartFiles {
	// 	srcFilePath := filepath.Join(jobTemplateDirPath, chartFile)
	// 	dstFilePath := filepath.Join(deploymentChartPath, chartFile)
	// 	err := util.CopyFile(srcFilePath, dstFilePath)
	// 	if err != nil {
	// 		errMsg := fmt.Sprintf("failed to copy a deployment chart file %s: %v", chartFile, err)
	// 		zap.S().Debugf(errMsg)

	// 		return fmt.Errorf(errMsg)
	// 	}
	// }

	// for _, taskInfo := range h.tasksInfo {
	// 	if taskInfo.Type == openapi.USER {
	// 		// don't attempt to provision compute resource for user-driven task
	// 		continue
	// 	}

	// 	context := map[string]string{
	// 		"imageLoc": h.jobParams.Image,
	// 		"taskId":   taskInfo.TaskId,
	// 		"taskKey":  taskInfo.Key,
	// 	}
	// 	rendered, err := mustache.RenderFile(jobTemplatePath, &context)
	// 	if err != nil {
	// 		errMsg := fmt.Sprintf("failed to render a template for task %s: %v", taskInfo.TaskId, err)
	// 		zap.S().Debugf(errMsg)

	// 		return fmt.Errorf(errMsg)
	// 	}

	// 	deploymentFileName := fmt.Sprintf("%s-%s.yaml", jobDeploymentFilePrefix, taskInfo.TaskId)
	// 	deploymentFilePath := filepath.Join(targetTemplateDirPath, deploymentFileName)
	// 	err = ioutil.WriteFile(deploymentFilePath, []byte(rendered), util.FilePerm0644)
	// 	if err != nil {
	// 		errMsg := fmt.Sprintf("failed to write a job rosource spec %s: %v", taskInfo.TaskId, err)
	// 		zap.S().Debugf(errMsg)

	// 		return fmt.Errorf(errMsg)
	// 	}
	// }

	// // TODO: when multiple clusters are supported,
	// //       set platform dynamically based on the target cluster type
	// dplyr, err := deployer.NewDeployer(h.platform)
	// if err != nil {
	// 	errMsg := fmt.Sprintf("failed to obtain a job deployer: %v", err)
	// 	zap.S().Debugf(errMsg)

	// 	return fmt.Errorf(errMsg)
	// }

	// err = dplyr.Initialize("", h.namespace)
	// if err != nil {
	// 	errMsg := fmt.Sprintf("failed to initialize a job deployer: %v", err)
	// 	zap.S().Debugf(errMsg)

	// 	return fmt.Errorf(errMsg)
	// }

	// err = dplyr.Install("job-"+h.jobId, deploymentChartPath)
	// if err != nil {
	// 	errMsg := fmt.Sprintf("failed to deploy tasks: %v", err)
	// 	zap.S().Debugf(errMsg)

	// 	return fmt.Errorf(errMsg)
	// }

	// h.dplyr = dplyr

	zap.S().Infof("Completed deployment of agents")
	return nil
}
