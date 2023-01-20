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
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/cisco-open/flame/cmd/controller/app/database"
	"github.com/cisco-open/flame/cmd/controller/config"
	"github.com/cisco-open/flame/pkg/openapi"
	pbNotify "github.com/cisco-open/flame/pkg/proto/notification"
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

	jobSpec openapi.JobSpec

	tasksInfo      []openapi.TaskInfo
	roles          []openapi.Role
	roleStateCount map[string]map[openapi.JobState]int

	tskEventCh       chan openapi.TaskInfo
	errCh            chan error
	tskWatchCancelFn context.CancelFunc

	// isDone is set in case where nothing can be done by a handler
	// isDone should be set only once as its buffer size is 1
	isDone chan bool

	state JobHandlerState

	bInsecure bool
	bPlain    bool
}

func NewDefaultHandler(dbService database.DBService, jobId string, userEventQ *EventQ,
	jobQueues map[string]*EventQ, mu *sync.Mutex, notifier string, jobParams config.JobParams,
	bInsecure bool, bPlain bool) (*DefaultHandler, error) {
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

		bInsecure: bInsecure,
		bPlain:    bPlain,

		tasksInfo: make([]openapi.TaskInfo, 0),
		roles:     make([]openapi.Role, 0),

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
		event := <-h.userEventQ.GetJobEventBuffer()
		event.ErrCh <- fmt.Errorf("failed to fetch job specification: %v", err)
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

		case event := <-h.userEventQ.GetJobEventBuffer():
			state := event.JobStatus.State
			if state != openapi.STARTING && state != openapi.STOPPING && state != openapi.APPLYING {
				event.ErrCh <- fmt.Errorf("status update operation not allowed from user")
				continue
			}
			h.doHandle(event)

		case event := <-h.sysEventQ.GetJobEventBuffer():
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
	err := h.notifyDeploy(pbNotify.DeployEventType_REVOKE_RESOURCE)
	zap.S().Infof("Invoked notifyDeploy - revoke resource for jobId %s", h.jobId)
	if err != nil {
		zap.S().Errorf("notifyDeploy could not notify all deployers to revoke resources, err: %v", err)
	} else {
		h.tskWatchCancelFn()
	}
}

func (h *DefaultHandler) ChangeState(state JobHandlerState) {
	h.state = state
}

func (h *DefaultHandler) allocateComputes() error {
	err := h.notifyDeploy(pbNotify.DeployEventType_ADD_RESOURCE)
	zap.S().Infof("Invoked notifyDeploy - add resource for jobId %s", h.jobId)
	if err != nil {
		zap.S().Errorf("notifyDeploy could not notify all deployers to allocate resources, err: %v", err)
		return err
	}
	return nil
}

func (h *DefaultHandler) notifyJob(evtType pbNotify.JobEventType) error {
	req := &pbNotify.JobEventRequest{
		Type:    evtType,
		TaskIds: make([]string, 0),
	}

	for _, taskInfo := range h.tasksInfo {
		req.JobId = taskInfo.JobId
		req.TaskIds = append(req.TaskIds, taskInfo.TaskId)
	}

	resp, err := newNotifyClient(h.notifier, h.bInsecure, h.bPlain).sendJobNotification(req)
	if err != nil {
		return fmt.Errorf("failed to notify for job: %v", err)
	}

	zap.S().Infof("response status from notifyJob = %s", resp.Status.String())

	return nil
}

func (h *DefaultHandler) notifyDeploy(evtType pbNotify.DeployEventType) error {
	// Get a list of computeIds to notify for deployment
	// To do this, get all tasks for the jobId, iterate and get computeId
	computeIdMap := make(map[string]bool)

	for _, taskInfo := range h.tasksInfo {
		taskCompute := taskInfo.ComputeId
		computeIdMap[taskCompute] = true
	}

	computeIds := make([]string, len(computeIdMap))
	i := 0
	for compute := range computeIdMap {
		computeIds[i] = compute
		i++
	}
	zap.S().Infof("In notifyDeploy for tasks, jobId %s, will be notifing computes: %v", h.jobId, computeIds)

	req := &pbNotify.DeployEventRequest{
		Type:       evtType,
		ComputeIds: computeIds,
		JobId:      h.jobId,
	}

	resp, err := newNotifyClient(h.notifier, h.bInsecure, h.bPlain).sendDeployNotification(req)
	if err != nil {
		return fmt.Errorf("failed to notify for deployment: %v, failed deployers: %v, status: %v", err, resp.FailedDeployers, resp.Status)
	}

	zap.S().Infof("response status from notifyDeploy = %s", resp.Status.String())

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
