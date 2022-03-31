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
	"fmt"

	"go.uber.org/zap"

	"github.com/cisco-open/flame/pkg/openapi"
	pbNotify "github.com/cisco-open/flame/pkg/proto/notification"
	"github.com/cisco-open/flame/pkg/util"
)

type JobHandlerState interface {
	ApplyChange()
	ApplyNone()
	CleanUp()
	Complete()
	Deploy(event *JobEvent)
	Fail()
	Run(event *JobEvent)
	Start(event *JobEvent)
	Stop(event *JobEvent)
	Timeout()
	Update(event *JobEvent)
}

type StateBase struct {
	hdlr *DefaultHandler
}

type StateReady struct {
	StateBase
}

type StateStarting struct {
	StateBase
}

type StateDeploying struct {
	StateBase
}

type StateRunning struct {
	StateBase
}

type StateStopping struct {
	StateBase
}

type StateCompleted struct {
	StateBase
}

type StateTerminated struct {
	StateBase
}

type StateFailed struct {
	StateBase
}

type StateApplying struct {
	StateBase
}

func NewStateReady(h *DefaultHandler) JobHandlerState {
	return &StateReady{StateBase{hdlr: h}}
}

func NewStateStarting(h *DefaultHandler) JobHandlerState {
	return &StateStarting{StateBase{hdlr: h}}
}

func NewStateDeploying(h *DefaultHandler) JobHandlerState {
	return &StateDeploying{StateBase{hdlr: h}}
}

func NewStateRunning(h *DefaultHandler) JobHandlerState {
	return &StateRunning{StateBase{hdlr: h}}
}

func NewStateStopping(h *DefaultHandler) JobHandlerState {
	return &StateStopping{StateBase{hdlr: h}}
}

func NewStateCompleted(h *DefaultHandler) JobHandlerState {
	return &StateCompleted{StateBase{hdlr: h}}
}

func NewStateTerminated(h *DefaultHandler) JobHandlerState {
	return &StateTerminated{StateBase{hdlr: h}}
}

func NewStateFailed(h *DefaultHandler) JobHandlerState {
	return &StateFailed{StateBase{hdlr: h}}
}

func NewStateApplying(h *DefaultHandler) JobHandlerState {
	return &StateApplying{StateBase{hdlr: h}}
}

///////////////////////////////////////////////////////////////////////////////
// The following implements default state handling methods, which do nothing.
// Each state struct only needs to implement functions that it cares about.

func (s *StateBase) ApplyChange() {
	zap.S().Info("Base ApplyChange called")
}

func (s *StateBase) ApplyNone() {
	zap.S().Info("Base ApplyNone called")
}

func (s *StateBase) CleanUp() {
	zap.S().Info("Base CleanUp called")
}

func (s *StateBase) Complete() {
	zap.S().Info("Base Complete called")
}

func (s *StateBase) Deploy(event *JobEvent) {
	zap.S().Info("Base Deploy called")
}

func (s *StateBase) Fail() {
	zap.S().Info("Base Fail called")
}

func (s *StateBase) Run(event *JobEvent) {
	zap.S().Info("Base Run called")
}

func (s *StateBase) Start(event *JobEvent) {
	zap.S().Info("Base Start called")
}

func (s *StateBase) Stop(event *JobEvent) {
	zap.S().Info("Base Stop called")
}

func (s *StateBase) Timeout() {
	zap.S().Info("Base Timeout called")
}

func (s *StateBase) Update(event *JobEvent) {
	zap.S().Info("Base Update called")
}

///////////////////////////////////////////////////////////////////////////////

func (s *StateReady) Start(event *JobEvent) {
	zap.S().Infof("requester: %s, jobId: %s", event.Requester, event.JobStatus.Id)

	// Get tasks info from DB; If failed, return error; Otherwise, proceed to the next step
	tasksInfo, err := s.hdlr.dbService.GetTasksInfo(event.Requester, event.JobStatus.Id, 0, true)
	if err != nil {
		event.ErrCh <- fmt.Errorf("failed to fetch tasks' info: %v", err)
		return
	}
	s.hdlr.tasksInfo = tasksInfo

	for _, taskInfo := range tasksInfo {
		_, ok := s.hdlr.roleStateCount[taskInfo.Role]
		if !ok {
			s.hdlr.roleStateCount[taskInfo.Role] = make(map[openapi.JobState]int)
		}
		s.hdlr.roleStateCount[taskInfo.Role][openapi.READY]++
	}

	schema, err := s.hdlr.dbService.GetDesignSchema(event.Requester, s.hdlr.jobSpec.DesignId, s.hdlr.jobSpec.SchemaVersion)
	if err != nil {
		event.ErrCh <- fmt.Errorf("failed to fetch schema: %v", err)
		return
	}
	s.hdlr.roles = schema.Roles

	// set the job state to STARTING
	err = s.hdlr.dbService.UpdateJobStatus(event.Requester, event.JobStatus.Id, event.JobStatus)
	if err != nil {
		event.ErrCh <- fmt.Errorf("failed to set job state to %s: %v", event.JobStatus.State, err)

		s.hdlr.isDone <- true
		return
	}

	// At this point, inform that there is no error by setting the error channel nil
	// This only means that the start-job request was accepted after basic check and
	// the job will be scheduled.
	event.ErrCh <- nil

	event.JobStatus.State = openapi.DEPLOYING
	s.hdlr.sysEventQ.Enqueue(event)
	s.hdlr.ChangeState(NewStateStarting(s.hdlr))
}

///////////////////////////////////////////////////////////////////////////////

func (s *StateStarting) Deploy(event *JobEvent) {
	setJobFailure := func() {
		event.JobStatus.State = openapi.FAILED
		s.hdlr.sysEventQ.Enqueue(event)
		s.hdlr.ChangeState(s)
	}

	// spin up compute resources
	err := s.hdlr.allocateComputes()
	if err != nil {
		zap.S().Debugf("Failed to allocate compute resources: %v", err)

		setJobFailure()
		return
	}

	// send a job start message to notifier
	err = s.hdlr.notify(pbNotify.EventType_START_JOB)
	if err != nil {
		zap.S().Debugf("%v", err)

		setJobFailure()
		return
	}

	event.JobStatus.State = openapi.DEPLOYING
	_ = s.hdlr.dbService.UpdateJobStatus(event.Requester, event.JobStatus.Id, event.JobStatus)

	s.hdlr.ChangeState(NewStateDeploying(s.hdlr))
}

func (s *StateStarting) Fail() {
	zap.S().Warnf("Job %s is failed at %s state", s.hdlr.jobId, openapi.STARTING)

	setFail(s.hdlr)
}

func (s *StateStarting) Timeout() {
	zap.S().Infof("Timeout triggered for job %s at %s state", s.hdlr.jobId, openapi.STARTING)

	setTimeout(s.hdlr)
}

///////////////////////////////////////////////////////////////////////////////

func (s *StateDeploying) Fail() {
	zap.S().Infof("Job %s is failed at %s state", s.hdlr.jobId, openapi.DEPLOYING)

	setFail(s.hdlr)
}

func (s *StateDeploying) Run(event *JobEvent) {
	zap.S().Infof("Run() for Job %s at %s state", s.hdlr.jobId, openapi.DEPLOYING)

	_ = s.hdlr.dbService.UpdateJobStatus(s.hdlr.jobSpec.UserId, s.hdlr.jobId, openapi.JobStatus{State: openapi.RUNNING})

	s.hdlr.ChangeState(NewStateRunning(s.hdlr))
}

func (s *StateDeploying) Stop(event *JobEvent) {
	zap.S().Warnf("%s state: Stop not yet implemented", openapi.DEPLOYING)
}

func (s *StateDeploying) Timeout() {
	zap.S().Infof("Timeout triggered for job %s at %s state", s.hdlr.jobId, openapi.DEPLOYING)

	setTimeout(s.hdlr)
}

///////////////////////////////////////////////////////////////////////////////

func (s *StateRunning) Complete() {
	zap.S().Infof("Job %s is completed", s.hdlr.jobId)

	// job is completed; update job state with COMPLETED
	_ = s.hdlr.dbService.UpdateJobStatus(s.hdlr.jobSpec.UserId, s.hdlr.jobId, openapi.JobStatus{State: openapi.COMPLETED})

	s.hdlr.ChangeState(NewStateCompleted(s.hdlr))

	s.hdlr.isDone <- true
}

func (s *StateRunning) Fail() {
	zap.S().Warnf("Job %s is failed at %s state", s.hdlr.jobId, openapi.RUNNING)

	setFail(s.hdlr)
}

func (s *StateRunning) Stop(event *JobEvent) {
	zap.S().Warnf("%s state: Stop not yet implemented", openapi.RUNNING)
}

func (s *StateRunning) Timeout() {
	zap.S().Infof("Timeout triggered for job %s at %s state", s.hdlr.jobId, openapi.RUNNING)

	setTimeout(s.hdlr)
}

func (s *StateRunning) Update(event *JobEvent) {
	zap.S().Warnf("%s state: Update not yet implemented", openapi.RUNNING)
}

///////////////////////////////////////////////////////////////////////////////

func (s *StateStopping) ApplyChange()           {}
func (s *StateStopping) ApplyNone()             {}
func (s *StateStopping) CleanUp()               {}
func (s *StateStopping) Complete()              {}
func (s *StateStopping) Deploy(event *JobEvent) {}
func (s *StateStopping) Fail()                  {}
func (s *StateStopping) Run(event *JobEvent)    {}
func (s *StateStopping) Start(event *JobEvent)  {}
func (s *StateStopping) Stop(event *JobEvent)   {}
func (s *StateStopping) Timeout()               {}
func (s *StateStopping) Update(event *JobEvent) {}

///////////////////////////////////////////////////////////////////////////////

func (s *StateApplying) ApplyChange()           {}
func (s *StateApplying) ApplyNone()             {}
func (s *StateApplying) CleanUp()               {}
func (s *StateApplying) Complete()              {}
func (s *StateApplying) Deploy(event *JobEvent) {}
func (s *StateApplying) Fail()                  {}
func (s *StateApplying) Run(event *JobEvent)    {}
func (s *StateApplying) Start(event *JobEvent)  {}
func (s *StateApplying) Stop(event *JobEvent)   {}
func (s *StateApplying) Timeout() {
	zap.S().Infof("Timeout triggered for job %s at %s state", s.hdlr.jobId, openapi.APPLYING)

	setTimeout(s.hdlr)
}

func (s *StateApplying) Update(event *JobEvent) {}

///////////////////////////////////////////////////////////////////////////////

func setFail(h *DefaultHandler) {
	setTaskStateToTerminated(h)

	_ = h.dbService.UpdateJobStatus(h.jobSpec.UserId, h.jobId, openapi.JobStatus{State: openapi.FAILED})

	h.ChangeState(NewStateFailed(h))

	h.isDone <- true
}

func setTimeout(h *DefaultHandler) {
	setTaskStateToTerminated(h)

	_ = h.dbService.UpdateJobStatus(h.jobSpec.UserId, h.jobId, openapi.JobStatus{State: openapi.TERMINATED})

	h.ChangeState(NewStateTerminated(h))

	h.isDone <- true
}

func setTaskStateToTerminated(h *DefaultHandler) {
	// For any tasks in running state and created by system,
	// set them terminated as they will be deleted by the system
	filter := map[string]interface{}{
		util.DBFieldState:    openapi.RUNNING,
		util.DBFieldTaskType: openapi.SYSTEM,
	}
	_ = h.dbService.UpdateTaskStateByFilter(h.jobId, openapi.TERMINATED, filter)
}
