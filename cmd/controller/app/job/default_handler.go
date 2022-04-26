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

package job

import (
	"context"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/cbroglie/mustache"
	"go.uber.org/zap"

	"github.com/cisco-open/flame/cmd/controller/app/database"
	"github.com/cisco-open/flame/cmd/controller/app/deployer"
	"github.com/cisco-open/flame/cmd/controller/config"
	"github.com/cisco-open/flame/pkg/openapi"
	pbNotify "github.com/cisco-open/flame/pkg/proto/notification"
	"github.com/cisco-open/flame/pkg/util"
)

type DefaultHandler struct {
	dbService  database.DBService
	jobId      string
	userEventQ *EventQ
	sysEventQ  *EventQ
	jobQueues  map[string]*EventQ
	mu         *sync.Mutex
	notifier   string
	jobParams  config.JobParams
	platform   string

	jobSpec openapi.JobSpec

	tasksInfo      []openapi.TaskInfo
	roles          []openapi.Role
	roleStateCount map[string]map[openapi.JobState]int

	tskEventCh       chan openapi.TaskInfo
	errCh            chan error
	tskWatchCancelFn context.CancelFunc

	dplyr deployer.Deployer
	// isDone is set in case where nothing can be done by a handler
	// isDone should be set only once as its buffer size is 1
	isDone chan bool

	state JobHandlerState
}

func NewDefaultHandler(dbService database.DBService, jobId string, userEventQ *EventQ, jobQueues map[string]*EventQ, mu *sync.Mutex,
	notifier string, jobParams config.JobParams, platform string) (*DefaultHandler, error) {
	// start task monitoring
	tskEventCh, errCh, tskWatchCancelFn, err := dbService.MonitorTasks(jobId)
	if err != nil {
		return nil, err
	}

	hdlr := &DefaultHandler{
		dbService:  dbService,
		jobId:      jobId,
		userEventQ: userEventQ,   // an event queue to receive event generated from user
		sysEventQ:  NewEventQ(0), // an event queue to used for transition between states
		jobQueues:  jobQueues,
		mu:         mu,
		notifier:   notifier,
		jobParams:  jobParams,
		platform:   platform,
		tasksInfo:  make([]openapi.TaskInfo, 0),
		roles:      make([]openapi.Role, 0),

		roleStateCount: make(map[string]map[openapi.JobState]int),

		tskEventCh:       tskEventCh,
		errCh:            errCh,
		tskWatchCancelFn: tskWatchCancelFn,

		isDone: make(chan bool, 1),
	}

	// Set state as ready to begin
	hdlr.state = NewStateReady(hdlr)

	return hdlr, nil
}

func (h *DefaultHandler) Do() {
	defer func() {
		h.mu.Lock()
		delete(h.jobQueues, h.jobId)
		h.mu.Unlock()
		zap.S().Infof("Cleaned up event queues for %s from job queues", h.jobId)
	}()

	jobSpec, err := h.dbService.GetJobById(h.jobId)
	if err != nil {
		zap.S().Errorf("Failed to fetch job specification: %v", err)
		return
	}
	h.jobSpec = jobSpec

	timer := time.NewTimer(time.Duration(h.jobSpec.MaxRunTime) * time.Second)
	for {
		select {
		case tskEvent := <-h.tskEventCh:
			h.handleTaskEvent(&tskEvent)

		case <-h.errCh:
			// non-fixable error occurred; halt job
			_ = h.dbService.UpdateJobStatus(h.jobSpec.UserId, h.jobId, openapi.JobStatus{State: openapi.FAILED})
			h.isDone <- true

		case <-timer.C:
			h.state.Timeout()

		case event := <-h.userEventQ.GetEventBuffer():
			state := event.JobStatus.State
			if state != openapi.STARTING && state != openapi.STOPPING && state != openapi.APPLYING {
				event.ErrCh <- fmt.Errorf("status update operation not allowed from user")
				continue
			}
			h.doHandle(event)

		case event := <-h.sysEventQ.GetEventBuffer():
			h.doHandle(event)

		case <-h.isDone:
			h.cleanup()
			return
		}
	}
}

func (h *DefaultHandler) doHandle(event *JobEvent) {
	switch event.JobStatus.State {
	// the following six states are managed/updated by controller internally
	// user cannot directly set these states
	case openapi.READY:
		event.ErrCh <- fmt.Errorf("state update operation not allowed")

	case openapi.STARTING:
		h.state.Start(event)

	case openapi.DEPLOYING:
		h.state.Deploy(event)

	case openapi.RUNNING:
		h.state.Run(event)

	case openapi.COMPLETED:
		h.state.Complete()

	case openapi.FAILED:
		h.state.Fail()

	case openapi.STOPPING:
		h.state.Stop(event)

	case openapi.TERMINATED:
		h.state.CleanUp()

	case openapi.APPLYING:
		h.state.Update(event)

	default:
		event.ErrCh <- fmt.Errorf("unknown state")
	}
}

func (h *DefaultHandler) cleanup() {
	// 1. decommission compute resources if they are in use
	if h.dplyr != nil {
		if err := h.dplyr.Uninstall("job-" + h.jobId); err != nil {
			zap.S().Warnf("failed to release resources for job %s: %v", h.jobId, err)
		}
	}

	// 2.delete all the job resource specification files
	deploymentChartPath := filepath.Join(deploymentDirPath, h.jobId)
	_ = os.RemoveAll(deploymentChartPath)

	h.tskWatchCancelFn()
}

func (h *DefaultHandler) ChangeState(state JobHandlerState) {
	h.state = state
}

func (h *DefaultHandler) allocateComputes() error {
	deploymentChartPath := filepath.Join(deploymentDirPath, h.jobId)
	targetTemplateDirPath := filepath.Join(deploymentChartPath, deploymentTemplateDir)
	if err := os.MkdirAll(targetTemplateDirPath, util.FilePerm0644); err != nil {
		errMsg := fmt.Sprintf("failed to create a deployment template folder: %v", err)
		zap.S().Debugf(errMsg)

		return fmt.Errorf(errMsg)
	}

	// Copy helm chart files to destination folder
	for _, chartFile := range helmChartFiles {
		srcFilePath := filepath.Join(deploymentDirPath, chartFile)
		dstFilePath := filepath.Join(deploymentChartPath, chartFile)
		err := util.CopyFile(srcFilePath, dstFilePath)
		if err != nil {
			errMsg := fmt.Sprintf("failed to copy a deployment chart file %s: %v", chartFile, err)
			zap.S().Debugf(errMsg)

			return fmt.Errorf(errMsg)
		}
	}

	for _, taskInfo := range h.tasksInfo {
		if taskInfo.Type == openapi.USER {
			// don't attempt to provision compute resource for user-driven task
			continue
		}

		context := map[string]string{
			"imageLoc": h.jobParams.Image,
			"taskId":   taskInfo.TaskId,
			"taskKey":  taskInfo.Key,
		}
		rendered, err := mustache.RenderFile(jobTemplatePath, &context)
		if err != nil {
			errMsg := fmt.Sprintf("failed to render a template for task %s: %v", taskInfo.TaskId, err)
			zap.S().Debugf(errMsg)

			return fmt.Errorf(errMsg)
		}

		deploymentFileName := fmt.Sprintf("%s-%s.yaml", jobDeploymentFilePrefix, taskInfo.TaskId)
		deploymentFilePath := filepath.Join(targetTemplateDirPath, deploymentFileName)
		err = ioutil.WriteFile(deploymentFilePath, []byte(rendered), util.FilePerm0644)
		if err != nil {
			errMsg := fmt.Sprintf("failed to write a job rosource spec %s: %v", taskInfo.TaskId, err)
			zap.S().Debugf(errMsg)

			return fmt.Errorf(errMsg)
		}
	}

	// TODO: when multiple clusters are supported,
	//       set platform dynamically based on the target cluster type
	dplyr, err := deployer.NewDeployer(h.platform)
	if err != nil {
		errMsg := fmt.Sprintf("failed to obtain a job deployer: %v", err)
		zap.S().Debugf(errMsg)

		return fmt.Errorf(errMsg)
	}

	err = dplyr.Initialize("", util.ProjectName)
	if err != nil {
		errMsg := fmt.Sprintf("failed to initialize a job deployer: %v", err)
		zap.S().Debugf(errMsg)

		return fmt.Errorf(errMsg)
	}

	err = dplyr.Install("job-"+h.jobId, deploymentChartPath)
	if err != nil {
		errMsg := fmt.Sprintf("failed to deploy tasks: %v", err)
		zap.S().Debugf(errMsg)

		return fmt.Errorf(errMsg)
	}

	h.dplyr = dplyr

	return nil
}

func (h *DefaultHandler) notify(evtType pbNotify.EventType) error {
	req := &pbNotify.EventRequest{
		Type:    evtType,
		TaskIds: make([]string, 0),
	}

	for _, taskInfo := range h.tasksInfo {
		req.JobId = taskInfo.JobId
		req.TaskIds = append(req.TaskIds, taskInfo.TaskId)
	}

	resp, err := newNotifyClient(h.notifier).sendNotification(req)
	if err != nil {
		return fmt.Errorf("failed to notify: %v", err)
	}

	zap.S().Infof("response status = %s", resp.Status.String())

	return nil
}

func (h *DefaultHandler) handleTaskEvent(tskEvent *openapi.TaskInfo) {
	if tskEvent != nil {
		_, ok := h.roleStateCount[tskEvent.Role]
		if ok {
			h.roleStateCount[tskEvent.Role][tskEvent.State]++
		}
	}

	if tskEvent.State == openapi.RUNNING {
		event := NewJobEvent("", openapi.JobStatus{State: openapi.RUNNING})
		h.sysEventQ.Enqueue(event)
	} else if h.isOneRoleAllFailed() {
		event := NewJobEvent("", openapi.JobStatus{State: openapi.FAILED})
		h.sysEventQ.Enqueue(event)
		return
	}

	if h.isOneTaskRunning() {
		// at least one task is in running state, there is nothing to do
		return
	}

	// at this point, none of tasks is in running state
	// check if at least one task in each role is completed
	if !h.isOneTaskCompletedPerRole() {
		// all tasks in some role were failed/terminated or never executed
		// so the job cannot be completed successfully
		event := NewJobEvent("", openapi.JobStatus{State: openapi.FAILED})
		h.sysEventQ.Enqueue(event)

		return
	}

	event := NewJobEvent("", openapi.JobStatus{State: openapi.COMPLETED})
	h.sysEventQ.Enqueue(event)
}

func (h *DefaultHandler) isOneRoleAllFailed() bool {
	for _, stateCount := range h.roleStateCount {
		if stateCount[openapi.READY] != stateCount[openapi.FAILED] {
			continue
		}

		// all tasks in some role were failed
		return true
	}

	return false
}

func (h *DefaultHandler) isOneTaskRunning() bool {
	for _, stateCount := range h.roleStateCount {
		if (stateCount[openapi.COMPLETED] + stateCount[openapi.TERMINATED] +
			stateCount[openapi.FAILED]) < stateCount[openapi.RUNNING] {
			return true
		}
	}

	return false
}

func (h *DefaultHandler) isOneTaskCompletedPerRole() bool {
	roleCount := len(h.roleStateCount)
	completedCount := 0
	for _, stateCount := range h.roleStateCount {
		if stateCount[openapi.COMPLETED] > 0 {
			completedCount++
		}
	}

	return roleCount == completedCount
}
