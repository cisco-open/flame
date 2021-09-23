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

package job

import (
	"fmt"
	"sync"

	"go.uber.org/zap"

	"wwwin-github.cisco.com/eti/fledge/cmd/controller/app/database"
	"wwwin-github.cisco.com/eti/fledge/pkg/openapi"
	pbNotify "wwwin-github.cisco.com/eti/fledge/pkg/proto/notification"
)

type Manager struct {
	jobEventQ *EventQ

	notifierEp string

	jobDoneMap sync.Map
}

func NewManager(jobEventQ *EventQ, notifierEp string) (*Manager, error) {
	if jobEventQ == nil {
		return nil, fmt.Errorf("job event queue is nil")
	}

	manager := &Manager{
		jobEventQ: jobEventQ,

		notifierEp: notifierEp,
	}

	return manager, nil
}

func (mgr *Manager) Do() {
	for {
		event := mgr.jobEventQ.Dequeue()

		_, ok := mgr.jobDoneMap.Load(event.JobStatus.Id)
		if ok {
			event.ErrCh <- fmt.Errorf("some other work in progress. try later.")
			continue
		}

		switch event.JobStatus.State {
		// the following six states are managed/updated by controller internally
		// user cannot directly set these states
		case openapi.READY:
			fallthrough
		case openapi.DEPLOYING:
			fallthrough
		case openapi.RUNNING:
			fallthrough
		case openapi.FAILED:
			fallthrough
		case openapi.TERMINATED:
			fallthrough
		case openapi.COMPLETED:
			event.ErrCh <- fmt.Errorf("status update operation not allowed")

		case openapi.STARTING:
			fallthrough
		case openapi.STOPPING:
			fallthrough
		case openapi.APPLYING:
			mgr.handleJob(event)

		default:
			event.ErrCh <- fmt.Errorf("unknown state")
		}
	}
}

func (mgr *Manager) handleJob(event *JobEvent) {
	// save the job ID to track if the request is handled or not
	mgr.jobDoneMap.Store(event.JobStatus.Id, false)

	if event.JobStatus.State == openapi.STARTING {
		go mgr.handleStart(event)
	} else if event.JobStatus.State == openapi.STOPPING {
		go mgr.handleStop(event)
	} else if event.JobStatus.State == openapi.APPLYING {
		go mgr.handleApply(event)
	} else {
		go mgr.handleUnexpected(event)
	}
}

func (mgr *Manager) handleStart(event *JobEvent) {
	// 1. check the current state of the job
	// 1-1. If the state is not ready / failed / terminated,
	//      return error with a message that a job is not in a scheduleable state
	// 1-2. Otherwise, proceed to the next step

	// 2. Generate payload (configuration and ML code) based on the job spec
	// 2-1. If payload generation failed, return error
	// 2-2. Otherwise, proceed to the next step

	// 3. update job state to STARTING in the db
	// 3-1. If successful, return nil error to user because it takes time to spin up compute resources
	// 3-2. Otherwise, return error

	// 4. spin up compute resources and handle remaining taaks.
	// 4-1. Once a minimum set of compute resources are provisioned (e.g., one compute node per role),
	//      set the state to RUNNING in the db.
	// 4-2. If the condition in 5-1 is not met for a certain duration, cancel the provisioning of
	//      the compute resources and destroy the provisioned compute resources; set the state to FAILED

	// 5. send start-job event to the agent (i.e., fledgelet) in all the provisioned compute nodes

	defer mgr.jobDoneMap.Delete(event.JobStatus.Id)

	zap.S().Infof("requester: %s, jobId: %s", event.Requester, event.JobStatus.Id)

	curStatus, err := database.GetJobStatus(event.Requester, event.JobStatus.Id)
	if err != nil {
		event.ErrCh <- fmt.Errorf("failed to check the current status: %v", err)
		return
	}

	if curStatus.State != openapi.READY && curStatus.State != openapi.FAILED && curStatus.State != openapi.COMPLETED {
		event.ErrCh <- fmt.Errorf("job not in a state to schedule")
		return
	}

	// Obtain job specification
	jobSpec, err := database.GetJob(event.Requester, event.JobStatus.Id)
	if err != nil {
		event.ErrCh <- fmt.Errorf("failed to fetch job spec")
		return
	}

	payloads, err := newJobBuilder(jobSpec).getPayloads()
	if err != nil {
		event.ErrCh <- fmt.Errorf("failed to generate payloads: %v", err)
		return
	}

	err = database.CreatePayloads(payloads)
	if err != nil {
		event.ErrCh <- err
		return
	}

	err = database.UpdateJobStatus(event.Requester, event.JobStatus.Id, event.JobStatus)
	if err != nil {
		event.ErrCh <- fmt.Errorf("failed to set job state to %s: %v", event.JobStatus.State, err)
		// TODO: need to rollback the changes
		return
	}

	// At this point, inform that there is no error by setting the error channel nil
	// This only means that the start-job request was accepted after basic check and
	// the job will be scheduled.
	event.ErrCh <- nil

	// TODO: spin up compute resources

	req := &pbNotify.EventRequest{
		Type:     pbNotify.EventType_START_JOB,
		AgentIds: make([]string, 0),
	}

	for _, payload := range payloads {
		req.JobId = payload.JobId
		req.AgentIds = append(req.AgentIds, payload.AgentId)
	}

	resp, err := newNotifyClient(mgr.notifierEp).sendNotification(req)
	if err != nil || resp.Status == pbNotify.Response_ERROR {
		event.JobStatus.State = openapi.FAILED
		_ = database.UpdateJobStatus(event.Requester, event.JobStatus.Id, event.JobStatus)
		// TODO: need to rollback the changes
		return
	}

	event.JobStatus.State = openapi.DEPLOYING
	_ = database.UpdateJobStatus(event.Requester, event.JobStatus.Id, event.JobStatus)
}

func (mgr *Manager) handleStop(event *JobEvent) {
	event.ErrCh <- fmt.Errorf("not yet implemented")

	defer mgr.jobDoneMap.Delete(event.JobStatus.Id)
}

func (mgr *Manager) handleApply(event *JobEvent) {
	event.ErrCh <- fmt.Errorf("not yet implemented")

	defer mgr.jobDoneMap.Delete(event.JobStatus.Id)
}

func (mgr *Manager) handleUnexpected(event *JobEvent) {
	event.ErrCh <- fmt.Errorf("can't process the state update request: %s", event.JobStatus.State)

	defer mgr.jobDoneMap.Delete(event.JobStatus.Id)
}
