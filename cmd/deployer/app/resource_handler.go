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
	"os"
	"path/filepath"
	"time"

	"github.com/cbroglie/mustache"
	"github.com/cenkalti/backoff/v4"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"

	"github.com/cisco-open/flame/cmd/deployer/app/deployer"
	"github.com/cisco-open/flame/cmd/deployer/config"
	"github.com/cisco-open/flame/pkg/openapi"
	"github.com/cisco-open/flame/pkg/openapi/constants"
	pbNotify "github.com/cisco-open/flame/pkg/proto/notification"
	"github.com/cisco-open/flame/pkg/restapi"
	"github.com/cisco-open/flame/pkg/util"
)

const (
	deploymentTemplateDir = "templates"

	k8sShortLabelLength = 12
)

var (
	helmChartFiles = []string{"Chart.yaml", "values.yaml"}
)

type resourceHandler struct {
	apiserverEp string
	notifierEp  string
	spec        openapi.ComputeSpec

	platform  string
	namespace string
	dplyr     deployer.Deployer

	// variables for job templates
	jobTemplateDirPath string
	jobTemplatePath    string
	deploymentDirPath  string

	stream pbNotify.DeployEventRoute_GetDeployEventClient

	grpcDialOpt grpc.DialOption
}

func NewResourceHandler(cfg *config.Config, computeSpec openapi.ComputeSpec, bInsecure bool, bPlain bool) *resourceHandler {
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

	dplyr, err := deployer.NewDeployer(cfg.Platform)
	if err != nil {
		zap.S().Errorf("failed to obtain a job deployer: %v", err)
		return nil
	}

	parentDir := filepath.Dir(cfg.JobTemplate.Folder)
	deploymentDirPath := filepath.Join(parentDir, "deployment")

	rHandler := &resourceHandler{
		apiserverEp: cfg.Apiserver,
		notifierEp:  cfg.Notifier,
		spec:        computeSpec,

		platform:  cfg.Platform,
		namespace: cfg.Namespace,
		dplyr:     dplyr,

		jobTemplateDirPath: cfg.JobTemplate.Folder,
		jobTemplatePath:    filepath.Join(cfg.JobTemplate.Folder, cfg.JobTemplate.File),
		deploymentDirPath:  deploymentDirPath,

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

		err = r.dealWith(resp)
		if err != nil {
			zap.S().Errorf("Failed to dealWith task: %v", err)
		}
	}

	zap.S().Info("Disconnected from notifier")
}

func (r *resourceHandler) dealWith(in *pbNotify.DeployEvent) error {
	switch in.GetType() {
	case pbNotify.DeployEventType_ADD_RESOURCE:
		err := r.addResource(in.JobId)
		if err != nil {
			// TODO propagate the error back to the control plane
			errMsg := fmt.Sprintf("deploy event addResource failed with err: %v", err)
			zap.S().Errorf(errMsg)
			return fmt.Errorf(errMsg)
		}

	case pbNotify.DeployEventType_REVOKE_RESOURCE:
		err := r.revokeResource(in.JobId)
		if err != nil {
			// TODO propagate the error back to the control plane
			errMsg := fmt.Sprintf("deploy event revokeResource failed with err: %v", err)
			zap.S().Errorf(errMsg)
			return fmt.Errorf(errMsg)
		}

	case pbNotify.DeployEventType_UNKNOWN_DEPLOYMENT_TYPE:
		fallthrough
	default:
		errMsg := fmt.Sprintf("invalid message type: %s", in.GetType())
		zap.S().Errorf(errMsg)
		return fmt.Errorf(errMsg)
	}
	return nil
}

func (r *resourceHandler) addResource(jobId string) error {
	zap.S().Infof("Received add resource request for job %s", jobId)

	// Sending request to apiserver to get deployment config for specific jobId and computeId
	deploymentConfig, err := r.getDeploymentConfig(jobId)
	// TODO update deployment status to computeCollection in DB via restapi
	if err != nil {
		zap.S().Errorf("Failed to get deploymentConfig for job %s: %v\n", jobId, err)
	}
	zap.S().Infof("Got deployment config from apiserver: %v", deploymentConfig)

	// Deploy resources (agents) for the job based on the configuration
	err = r.deployResources(deploymentConfig)
	if err != nil {
		zap.S().Errorf("Failed to deploy resources for job %s: %v\n", jobId, err)
		return err
	}
	zap.S().Infof("Successfully added resources for compute %s and jobId %s", r.spec.ComputeId, jobId)
	return nil
}

func (r *resourceHandler) revokeResource(jobId string) (err error) {
	zap.S().Infof("Received revoke resource request for job %s", jobId)
	var cfg openapi.DeploymentConfig
	cfg, err = r.getDeploymentConfig(jobId)
	if err != nil {
		zap.S().Errorf("Failed to get %s job configs", jobId)
		return fmt.Errorf("failed to revoke %s job: %w", jobId, err)
	}
	taskStatuses := map[string]openapi.AgentState{}
	defer r.postDeploymentStatus(jobId, taskStatuses)
	for taskId := range cfg.AgentKVs {
		// 1. decommission compute resources one by one if they are in use
		if r.dplyr != nil {
			uninstallErr := r.dplyr.Uninstall("job-" + jobId + "-" + taskId[:k8sShortLabelLength])
			if uninstallErr != nil {
				zap.S().Warnf("failed to release resources for job %s: %v", jobId, uninstallErr)
				err = fmt.Errorf("%v; %v", err, uninstallErr)
				taskStatuses[taskId] = openapi.AGENT_REVOKE_FAILED
				continue
			}
			taskStatuses[taskId] = openapi.AGENT_REVOKE_SUCCESS
			// 2.delete all the task resource specification files
			deploymentChartPath := filepath.Join(r.deploymentDirPath, jobId, taskId)
			removeErr := os.RemoveAll(deploymentChartPath)
			if removeErr != nil {
				zap.S().Errorf("Errors occurred deleting specification files: %v", removeErr)
				// not returning error as these files doesn't affect system much.
				continue
			}
		}
	}
	zap.S().Infof("Completed revocation of agent resources")
	return err
}

func (r *resourceHandler) postDeploymentStatus(jobId string, taskStatuses map[string]openapi.AgentState) {
	if len(taskStatuses) == 0 {
		zap.S().Debugf("Skipped posting deployment status as no agent status received")
		return
	}
	zap.S().Infof("Posting deployment status (%v), for computeid: %s, for jobid: %s ",
		taskStatuses, r.spec.ComputeId, jobId)
	// construct url
	uriMap := map[string]string{
		constants.ParamComputeID: r.spec.ComputeId,
		constants.ParamJobID:     jobId,
	}
	url := restapi.CreateURL(r.apiserverEp, restapi.PutDeploymentStatusEndpoint, uriMap)

	code, _, err := restapi.HTTPPut(url, taskStatuses, "application/json; charset=utf-8")
	if err != nil || restapi.CheckStatusCode(code) != nil {
		zap.S().Errorf("Deployer failed to post deployment status %s for job %s - code: %d, error: %v\n",
			taskStatuses, jobId, code, err)
		return
	}
}

func (r *resourceHandler) getDeploymentConfig(jobId string) (openapi.DeploymentConfig, error) {
	zap.S().Infof("Sending request to apiserver / controller to get deployment config")
	// construct url
	uriMap := map[string]string{
		constants.ParamComputeID: r.spec.ComputeId,
		constants.ParamJobID:     jobId,
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

func copyHelmCharts(chartFiles []string, src, dst string) error {
	for _, chartFile := range chartFiles {
		srcFilePath := filepath.Join(src, chartFile)
		dstFilePath := filepath.Join(dst, chartFile)
		if err := util.CopyFile(srcFilePath, dstFilePath); err != nil {
			return fmt.Errorf("failed to copy a deployment chart file %s: %v", chartFile, err)
		}
	}
	return nil
}

func (r *resourceHandler) deployResources(deploymentConfig openapi.DeploymentConfig) (err error) {
	if err = r.dplyr.Initialize("", r.namespace); err != nil {
		errMsg := fmt.Sprintf("failed to initialize a job deployer: %v", err)
		return fmt.Errorf(errMsg)
	}

	agentStatuses := map[string]openapi.AgentState{}
	defer r.postDeploymentStatus(deploymentConfig.JobId, agentStatuses)

	for taskId := range deploymentConfig.AgentKVs {
		deploymentChartPath := filepath.Join(r.deploymentDirPath, deploymentConfig.JobId, taskId)
		targetTemplateDirPath := filepath.Join(deploymentChartPath, deploymentTemplateDir)

		if makeErr := os.MkdirAll(targetTemplateDirPath, util.FilePerm0644); makeErr != nil {
			errMsg := fmt.Sprintf("failed to create a deployment template folder: %v", makeErr)
			err = fmt.Errorf("%v; %v", err, errMsg)
			agentStatuses[taskId] = openapi.AGENT_DEPLOY_FAILED
			continue
		}

		// Copy helm chart files to destination folder
		copyErr := copyHelmCharts(helmChartFiles, r.jobTemplateDirPath, deploymentChartPath)
		if copyErr != nil {
			err = fmt.Errorf("%v; %v", err, copyErr)
			agentStatuses[taskId] = openapi.AGENT_DEPLOY_FAILED
			continue
		}

		ctx := map[string]string{
			"imageLoc":            deploymentConfig.ImageLoc,
			constants.ParamTaskID: taskId,
			"taskKey":             deploymentConfig.AgentKVs[taskId],
		}

		rendered, renderErr := mustache.RenderFile(r.jobTemplatePath, &ctx)
		if renderErr != nil {
			errMsg := fmt.Sprintf("failed to render a template for task %s: %v", taskId, renderErr)
			err = fmt.Errorf("%v; %v", err, errMsg)
			agentStatuses[taskId] = openapi.AGENT_DEPLOY_FAILED
			continue
		}

		deploymentFileName := fmt.Sprintf("task-%s.yaml", taskId)
		deploymentFilePath := filepath.Join(targetTemplateDirPath, deploymentFileName)

		writeErr := os.WriteFile(deploymentFilePath, []byte(rendered), util.FilePerm0644)
		if writeErr != nil {
			errMsg := fmt.Sprintf("failed to write a job rosource spec %s: %v", taskId, writeErr)
			err = fmt.Errorf("%v; %v", err, errMsg)
			agentStatuses[taskId] = openapi.AGENT_DEPLOY_FAILED
			continue
		}

		//using short id of task as label name does not support more than 35 characters
		installErr := r.dplyr.Install("job-"+deploymentConfig.JobId+"-"+taskId[:k8sShortLabelLength], deploymentChartPath)
		if installErr != nil {
			errMsg := fmt.Sprintf("failed to deploy tasks: %v", installErr)
			err = fmt.Errorf("%v; %v", err, errMsg)
			agentStatuses[taskId] = openapi.AGENT_DEPLOY_FAILED
			continue
		}

		agentStatuses[taskId] = openapi.AGENT_DEPLOY_SUCCESS
	}

	return err
}
