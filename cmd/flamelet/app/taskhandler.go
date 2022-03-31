// Copyright (c) 2021 Cisco Systems, Inc. and its affiliates
// All rights reserved
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package app

import (
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	backoff "github.com/cenkalti/backoff/v4"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"github.com/cisco-open/flame/pkg/openapi"
	pbNotify "github.com/cisco-open/flame/pkg/proto/notification"
	"github.com/cisco-open/flame/pkg/restapi"
	"github.com/cisco-open/flame/pkg/util"
)

const (
	workDir       = "/flame/work"
	pythonBin     = "python3"
	taskPyFile    = "main.py"
	logFilePrefix = "task"
	logFileExt    = "log"

	keyAgentId = "agentid"
	keyRole    = "role"
)

type taskHandler struct {
	apiserverEp string
	notifierEp  string
	name        string
	jobId       string
	agentId     string
	agentKey    string

	stream pbNotify.EventRoute_GetEventClient

	// role of the task given to flamelet
	role string

	state  openapi.JobState
	cancel context.CancelFunc
}

func newTaskHandler(apiserverEp string, notifierEp string, name string, agentId string, agentKey string) *taskHandler {
	return &taskHandler{
		apiserverEp: apiserverEp,
		notifierEp:  notifierEp,
		name:        name,
		agentId:     agentId,
		agentKey:    agentKey,
		state:       openapi.READY,
	}
}

// start connects to the notifier via grpc and handles notifications from the notifier
func (t *taskHandler) start() {
	go t.doStart()
}

func (t *taskHandler) doStart() {
	pauseTime := 10 * time.Second

	for {
		expBackoff := backoff.NewExponentialBackOff()
		expBackoff.MaxElapsedTime = 5 * time.Minute // max wait time: 5 minutes
		err := backoff.Retry(t.connect, expBackoff)
		if err != nil {
			zap.S().Fatalf("Cannot connect with notifier: %v", err)
		}

		t.do()

		// if connection is broken right after connection is made, this can cause
		// too many connection/disconnection events. To migitage that, add some static
		// pause time.
		time.Sleep(pauseTime)
	}
}

func (t *taskHandler) connect() error {
	// dial server
	conn, err := grpc.Dial(t.notifierEp, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		zap.S().Debugf("Cannot connect with notifier: %v", err)
		return err
	}

	client := pbNotify.NewEventRouteClient(conn)
	in := &pbNotify.AgentInfo{
		Id:       t.agentId,
		Hostname: t.name,
	}

	// setup notification stream
	stream, err := client.GetEvent(context.Background(), in)
	if err != nil {
		zap.S().Debugf("Open stream error: %v", err)
		return err
	}

	t.stream = stream
	zap.S().Infof("Connected with notifier at %s", t.notifierEp)

	return nil
}

func (t *taskHandler) do() {
	for {
		resp, err := t.stream.Recv()
		if err != nil {
			zap.S().Errorf("Failed to receive notification: %v", err)
			break
		}

		t.dealWith(resp)
	}

	zap.S().Info("Disconnected from notifier")
}

//newNotification acts as a handler and calls respective functions based on the response type to act on the received notifications.
func (t *taskHandler) dealWith(in *pbNotify.Event) {
	switch in.GetType() {
	case pbNotify.EventType_START_JOB:
		t.startJob(in.JobId)

	case pbNotify.EventType_STOP_JOB:
		t.stopJob(in.JobId)

	case pbNotify.EventType_UPDATE_JOB:
		t.updateJob(in.JobId)

	case pbNotify.EventType_UNKNOWN_EVENT_TYPE:
		fallthrough
	default:
		zap.S().Errorf("Invalid message type: %s", in.GetType())
	}
}

// startJob starts the application on the agent
func (t *taskHandler) startJob(jobId string) {
	zap.S().Infof("Received start job request on job %s", jobId)

	if t.state == openapi.RUNNING {
		zap.S().Infof("Task is already running job %s", jobId)
		return
	}
	// assign job ID to task handler
	t.jobId = jobId

	filePaths, err := t.getTask()
	if err != nil {
		zap.S().Warnf("Failed to download payload: %v", err)
		return
	}

	err = t.prepareTask(filePaths)
	if err != nil {
		zap.S().Warnf("Failed to prepare task")
		return
	}

	go t.runTask()

	// TODO: implement updateTaskStatus method
}

func (t *taskHandler) getTask() ([]string, error) {
	// construct URL
	uriMap := map[string]string{
		"jobId":   t.jobId,
		"agentId": t.agentId,
		"key":     t.agentKey,
	}
	url := restapi.CreateURL(t.apiserverEp, restapi.GetTaskEndpoint, uriMap)

	code, taskMap, err := restapi.HTTPGetMultipart(url)
	if err != nil || restapi.CheckStatusCode(code) != nil {
		errMsg := fmt.Sprintf("Failed to fetch task - code: %d, error: %v", code, err)
		zap.S().Warnf(errMsg)
		return nil, fmt.Errorf(errMsg)
	}

	filePaths := make([]string, 0)
	for fileName, data := range taskMap {
		filePath := filepath.Join("/tmp", fileName)
		err = ioutil.WriteFile(filePath, data, util.FilePerm0755)
		if err != nil {
			zap.S().Warnf("Failed to save %s: %v\n", fileName, err)
			return nil, err
		}

		filePaths = append(filePaths, filePath)

		zap.S().Infof("Downloaded %s successfully", fileName)
	}

	return filePaths, nil
}

func (t *taskHandler) prepareTask(filePaths []string) error {
	err := os.MkdirAll(workDir, util.FilePerm0755)
	if err != nil {
		return err
	}

	var fileDataList []util.FileData
	var file *os.File
	configFilePath := ""
	configFound := false
	codeFound := false
	for _, filePath := range filePaths {
		if strings.Contains(filePath, util.TaskConfigFile) {
			configFound = true

			configFilePath = filePath
		} else if strings.Contains(filePath, util.TaskCodeFile) {
			codeFound = true

			file, err = os.Open(filePath)
			if err != nil {
				return fmt.Errorf("failed to open %s: %v", filePath, err)
			}

			fileDataList, err = util.UnzipFile(file)
			if err != nil {
				return err
			}
		}
	}

	if !configFound || !codeFound {
		return fmt.Errorf("either %s or %s not found", util.TaskConfigFile, util.TaskCodeFile)
	}

	err = t.prepareConfig(configFilePath)
	if err != nil {
		return err
	}

	err = t.prepareCode(fileDataList)
	if err != nil {
		return err
	}

	return nil
}

func (t *taskHandler) prepareConfig(configFilePath string) error {
	// copy config file to work directory
	input, err := ioutil.ReadFile(configFilePath)
	if err != nil {
		return fmt.Errorf("failed to open config file %s: %v", configFilePath, err)
	}

	tmp := make(map[string]interface{})
	err = json.Unmarshal(input, &tmp)
	if err != nil {
		return fmt.Errorf("failed to unmarhsal config data")
	}
	t.role = tmp[keyRole].(string)

	// add agent id in the config
	tmp[keyAgentId] = t.agentId

	// marshall the updated config
	input, err = json.Marshal(&tmp)
	if err != nil {
		return fmt.Errorf("failed to marhsal config data")
	}

	dstFilePath := filepath.Join(workDir, util.TaskConfigFile)
	err = ioutil.WriteFile(dstFilePath, input, util.FilePerm0644)
	if err != nil {
		return fmt.Errorf("failed to copy config file: %v", err)
	}

	return nil
}

func (t *taskHandler) prepareCode(fileDataList []util.FileData) error {
	// copy code files to work directory
	for _, fileData := range fileDataList {
		dirPath := filepath.Join(workDir, filepath.Dir(fileData.FullName))
		err := os.MkdirAll(dirPath, util.FilePerm0755)
		if err != nil {
			return fmt.Errorf("failed to create directory: %v", err)
		}

		filePath := filepath.Join(dirPath, fileData.BaseName)
		err = ioutil.WriteFile(filePath, []byte(fileData.Data), util.FilePerm0644)
		if err != nil {
			return fmt.Errorf("failed to unzip file %s: %v", filePath, err)
		}
	}

	return nil
}

func (t *taskHandler) stopJob(jobId string) {
	if t.jobId != jobId {
		zap.S().Warnf("stop request on a wrong job %s", jobId)
		return
	}

	if t.state != openapi.RUNNING {
		zap.S().Warnf("job  %s is not in a running state", jobId)
		return
	}

	if t.cancel == nil {
		zap.S().Warnf("cancel function not specified for job %s", jobId)
		return
	}

	t.cancel()
}

func (t *taskHandler) updateJob(jobId string) (string, error) {
	zap.S().Infof("not yet implemented; received update job request on job %s", jobId)
	return "", nil
}

func (t *taskHandler) runTask() {
	taskFilePath := filepath.Join(workDir, t.role, taskPyFile)
	configFilePath := filepath.Join(workDir, util.TaskConfigFile)

	ctx, cacnel := context.WithCancel(context.Background())
	defer cacnel()
	t.cancel = cacnel

	// TODO: run the task in different user group with less privilege
	cmd := exec.CommandContext(ctx, pythonBin, taskFilePath, configFilePath)
	zap.S().Debugf("Running task with command: %v", cmd)

	logFileName := fmt.Sprintf("%s-%s.%s", logFilePrefix, t.jobId, logFileExt)
	logFilePath := filepath.Join(util.LogDirPath, logFileName)
	file, err := os.Create(logFilePath)
	if err != nil {
		zap.S().Errorf("Failed to create a log file: %v", err)
		return
	}
	defer file.Close()

	cmd.Stdout = file
	cmd.Stderr = file

	zap.S().Infof("Starting task for job %s", t.jobId)
	// set running state in advance
	// this is for keeping state transition simple, meaning that flamelet always
	// set running state first before making transition to one of three states
	// -- failed, terminated, completed
	t.updateTaskStatus(openapi.RUNNING)

	err = cmd.Start()
	if err != nil {
		zap.S().Errorf("Failed to start task: %v", err)
		t.updateTaskStatus(openapi.FAILED)
		return
	}

	err = cmd.Wait()
	if err != nil {
		if ctx.Err() == context.Canceled {
			// cancel() function was called
			zap.S().Infof("Task execution canceled for job %s: %v", t.jobId, err)
			t.updateTaskStatus(openapi.TERMINATED)
		} else {
			zap.S().Infof("Task execution failed for job %s: %v", t.jobId, err)
			t.updateTaskStatus(openapi.FAILED)
		}
	} else {
		zap.S().Infof("Task execution successful for job %s", t.jobId)
		t.updateTaskStatus(openapi.COMPLETED)
	}
}

func (t *taskHandler) updateTaskStatus(state openapi.JobState) {
	// update state in the task handler
	t.state = state

	// construct URL
	uriMap := map[string]string{
		"jobId":   t.jobId,
		"agentId": t.agentId,
	}
	url := restapi.CreateURL(t.apiserverEp, restapi.UpdateTaskStatusEndPoint, uriMap)

	taskStatus := openapi.TaskStatus{
		State: state,
	}

	code, _, err := restapi.HTTPPut(url, taskStatus, "application/json")
	if err != nil || restapi.CheckStatusCode(code) != nil {
		zap.S().Warnf("Failed to update a task status - code: %d, error: %v\n", code, err)
		return
	}
}
