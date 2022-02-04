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
	"context"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/cbroglie/mustache"
	"go.uber.org/zap"

	"github.com/cisco/fledge/cmd/controller/app/database"
	"github.com/cisco/fledge/cmd/controller/app/deployer"
	"github.com/cisco/fledge/cmd/controller/config"
	"github.com/cisco/fledge/pkg/openapi"
	pbNotify "github.com/cisco/fledge/pkg/proto/notification"
	"github.com/cisco/fledge/pkg/util"
)

const (
	deploymentDirPath     = "/" + util.ProjectName + "/deployment"
	deploymentTemplateDir = "templates"

	jobDeploymentFilePrefix = "job-agent"
	jobTemplatePath         = deploymentDirPath + "/" + jobDeploymentFilePrefix + ".yaml.mustache"
)

var (
	helmChartFiles = []string{"Chart.yaml", "values.yaml"}
)

type handler struct {
	dbService database.DBService
	jobId     string
	eventQ    *EventQ
	jobQueues map[string]*EventQ
	mu        *sync.Mutex
	notifier  string
	jobParams config.JobParams
	platform  string

	jobSpec openapi.JobSpec

	startTime time.Time

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
}

func NewHandler(dbService database.DBService, jobId string, eventQ *EventQ, jobQueues map[string]*EventQ, mu *sync.Mutex,
	notifier string, jobParams config.JobParams, platform string) (*handler, error) {
	// start task monitoring
	tskEventCh, errCh, tskWatchCancelFn, err := dbService.MonitorTasks(jobId)
	if err != nil {
		return nil, err
	}

	hdlr := &handler{
		dbService: dbService,
		jobId:     jobId,
		eventQ:    eventQ,
		jobQueues: jobQueues,
		mu:        mu,
		notifier:  notifier,
		jobParams: jobParams,
		platform:  platform,
		tasksInfo: make([]openapi.TaskInfo, 0),
		roles:     make([]openapi.Role, 0),

		roleStateCount: make(map[string]map[openapi.JobState]int),

		tskEventCh:       tskEventCh,
		errCh:            errCh,
		tskWatchCancelFn: tskWatchCancelFn,

		isDone: make(chan bool, 1),
	}

	return hdlr, nil
}

func (h *handler) Do() {
	defer func() {
		h.mu.Lock()
		delete(h.jobQueues, h.jobId)
		h.mu.Unlock()
		zap.S().Infof("Deleted an eventQ for %s from job queues", h.jobId)
	}()

	jobSpec, err := h.dbService.GetJobById(h.jobId)
	if err != nil {
		zap.S().Errorf("Failed to fetch job specification: %v", err)
		return
	}
	h.jobSpec = jobSpec

	h.startTime = time.Now()
	for {
		select {
		case tskEvent := <-h.tskEventCh:
			h.checkProgress(&tskEvent)

		case <-h.errCh:
			// non-fixable error occurred; halt job
			_ = h.dbService.UpdateJobStatus(h.jobSpec.UserId, h.jobId, openapi.JobStatus{State: openapi.FAILED})
			h.isDone <- true

		case event := <-h.eventQ.GetEventBuffer():
			h.doHandle(event)

		case <-h.isDone:
			h.cleanup()
			return
		}
	}
}

// checkProgress checks the progress of a job and sets a boolean flag (isDone) to tell if the job handling thread can be finished.
// If the job is in completed/failed/terminated state, this method sets true (meaning that the thread can be removed).
// It also set the flag true if the maximum wait time exceeds. Otherwise, it leaves the flag as is.
func (h *handler) checkProgress(tskEvent *openapi.TaskInfo) {
	// if the job didn't finish after max wait time, we finish the job handling/monitoring work
	if h.startTime.Add(time.Duration(h.jobSpec.MaxRunTime) * time.Second).Before(time.Now()) {
		zap.S().Infof("Job timed out")

		_ = h.dbService.UpdateJobStatus(h.jobSpec.UserId, h.jobId, openapi.JobStatus{State: openapi.TERMINATED})

		h.isDone <- true
		return
	}

	if tskEvent != nil {
		_, ok := h.roleStateCount[tskEvent.Role]
		if ok {
			h.roleStateCount[tskEvent.Role][tskEvent.State]++
		}
	}

	jobStatus, err := h.dbService.GetJobStatus(h.jobSpec.UserId, h.jobId)
	if err != nil {
		zap.S().Warnf("failed to get job status: %v", err)
		return
	}

	if jobStatus.State == openapi.DEPLOYING {
		if tskEvent.State == openapi.RUNNING {
			_ = h.dbService.UpdateJobStatus(h.jobSpec.UserId, h.jobId, openapi.JobStatus{State: openapi.RUNNING})
		} else if h.isOneRoleAllFailed() {
			zap.S().Warnf("Job %s is failed at %s state", h.jobId, jobStatus.State)
			_ = h.dbService.UpdateJobStatus(h.jobSpec.UserId, h.jobId, openapi.JobStatus{State: openapi.FAILED})
			h.isDone <- true
		}
	} else if jobStatus.State == openapi.RUNNING {
		if h.isOneTaskRunning() {
			// at least one task is in running state, there is nothing to do
			return
		}
		// at this point, none of tasks is in running state

		// check if at least one task in each role is completed
		if !h.isOneTaskCompletedPerRole() {
			// all tasks in some role were failed/terminated or never executed
			// so the job cannot be completed successfully
			zap.S().Warnf("Job %s is failed at %s state", h.jobId, jobStatus.State)
			_ = h.dbService.UpdateJobStatus(h.jobSpec.UserId, h.jobId, openapi.JobStatus{State: openapi.FAILED})
			h.isDone <- true
			return
		}

		zap.S().Infof("Job %s is completed", h.jobId)
		// job is completed; update job state with COMPLETED
		_ = h.dbService.UpdateJobStatus(h.jobSpec.UserId, h.jobId, openapi.JobStatus{State: openapi.COMPLETED})

		h.isDone <- true
	}
}

func (h *handler) isOneRoleAllFailed() bool {
	for _, stateCount := range h.roleStateCount {
		if stateCount[openapi.READY] != stateCount[openapi.FAILED] {
			continue
		}

		// all tasks in some role were failed
		return true
	}

	return false
}

func (h *handler) isOneTaskRunning() bool {
	for _, stateCount := range h.roleStateCount {
		if (stateCount[openapi.COMPLETED] + stateCount[openapi.TERMINATED] +
			stateCount[openapi.FAILED]) < stateCount[openapi.RUNNING] {
			return true
		}
	}

	return false
}

func (h *handler) isOneTaskCompletedPerRole() bool {
	roleCount := len(h.roleStateCount)
	completedCount := 0
	for _, stateCount := range h.roleStateCount {
		if stateCount[openapi.COMPLETED] > 0 {
			completedCount++
		}
	}

	return roleCount == completedCount
}

func (h *handler) doHandle(event *JobEvent) {
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

	// user is allowed to specify the following three states only:
	case openapi.STARTING:
		h.handleStart(event)

	case openapi.STOPPING:
		h.handleStop(event)

	case openapi.APPLYING:
		h.handleApply(event)

	default:
		event.ErrCh <- fmt.Errorf("unknown state")
	}
}

func (h *handler) handleStart(event *JobEvent) {
	// 1. check the current state of the job
	// 1-1. If the state is not ready / failed / terminated,
	//      return error with a message that a job is not in a scheduleable state
	// 1-2. Otherwise, proceed to the next step

	// 2. Get tasks info from DB
	// 2-1. If failed, return error
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

	zap.S().Infof("requester: %s, jobId: %s", event.Requester, event.JobStatus.Id)

	curStatus, err := h.dbService.GetJobStatus(event.Requester, event.JobStatus.Id)
	if err != nil {
		event.ErrCh <- fmt.Errorf("failed to check the current status: %v", err)
		return
	}

	if !(curStatus.State == openapi.COMPLETED || curStatus.State == openapi.FAILED ||
		curStatus.State == openapi.TERMINATED || curStatus.State == openapi.READY) {
		event.ErrCh <- fmt.Errorf("job not in a state to schedule")
		return
	}

	tasksInfo, err := h.dbService.GetTasksInfo(event.Requester, event.JobStatus.Id, 0, true)
	if err != nil {
		event.ErrCh <- fmt.Errorf("failed to fetch tasks' info: %v", err)
		return
	}
	h.tasksInfo = tasksInfo
	for _, taskInfo := range tasksInfo {
		_, ok := h.roleStateCount[taskInfo.Role]
		if !ok {
			h.roleStateCount[taskInfo.Role] = make(map[openapi.JobState]int)
		}
		h.roleStateCount[taskInfo.Role][taskInfo.State] = 0
	}

	schema, err := h.dbService.GetDesignSchema(event.Requester, h.jobSpec.DesignId, h.jobSpec.SchemaVersion)
	if err != nil {
		event.ErrCh <- fmt.Errorf("failed to fetch schema: %v", err)
		return
	}
	h.roles = schema.Roles

	// set the job state to STARTING
	err = h.dbService.UpdateJobStatus(event.Requester, event.JobStatus.Id, event.JobStatus)
	if err != nil {
		event.ErrCh <- fmt.Errorf("failed to set job state to %s: %v", event.JobStatus.State, err)

		h.isDone <- true
		return
	}

	// At this point, inform that there is no error by setting the error channel nil
	// This only means that the start-job request was accepted after basic check and
	// the job will be scheduled.
	event.ErrCh <- nil

	setJobFailure := func() {
		event.JobStatus.State = openapi.FAILED
		_ = h.dbService.UpdateJobStatus(event.Requester, event.JobStatus.Id, event.JobStatus)

		h.isDone <- true
	}

	err = h.allocateComputes()
	if err != nil {
		zap.S().Debugf("Failed to allocate compute resources: %v", err)

		setJobFailure()
		return
	}

	err = h.notify(pbNotify.EventType_START_JOB)
	if err != nil {
		zap.S().Debugf("%v", err)

		setJobFailure()
		return
	}

	event.JobStatus.State = openapi.DEPLOYING
	_ = h.dbService.UpdateJobStatus(event.Requester, event.JobStatus.Id, event.JobStatus)
}

func (h *handler) handleStop(event *JobEvent) {
	event.ErrCh <- fmt.Errorf("not yet implemented")
}

func (h *handler) handleApply(event *JobEvent) {
	event.ErrCh <- fmt.Errorf("not yet implemented")
}

func (h *handler) cleanup() {
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

func (h *handler) notify(evtType pbNotify.EventType) error {
	req := &pbNotify.EventRequest{
		Type:     evtType,
		AgentIds: make([]string, 0),
	}

	for _, taskInfo := range h.tasksInfo {
		req.JobId = taskInfo.JobId
		req.AgentIds = append(req.AgentIds, taskInfo.AgentId)
	}

	resp, err := newNotifyClient(h.notifier).sendNotification(req)
	if err != nil {
		return fmt.Errorf("failed to notify: %v", err)
	}

	zap.S().Infof("response status = %s", resp.Status.String())

	return nil
}

func (h *handler) allocateComputes() error {
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
			"agentId":  taskInfo.AgentId,
			"agentKey": taskInfo.Key,
		}
		rendered, err := mustache.RenderFile(jobTemplatePath, &context)
		if err != nil {
			errMsg := fmt.Sprintf("failed to render a template for agent %s: %v", taskInfo.AgentId, err)
			zap.S().Debugf(errMsg)

			return fmt.Errorf(errMsg)
		}

		deploymentFileName := fmt.Sprintf("%s-%s.yaml", jobDeploymentFilePrefix, taskInfo.AgentId)
		deploymentFilePath := filepath.Join(targetTemplateDirPath, deploymentFileName)
		err = ioutil.WriteFile(deploymentFilePath, []byte(rendered), util.FilePerm0644)
		if err != nil {
			errMsg := fmt.Sprintf("failed to write a job rosource spec %s: %v", taskInfo.AgentId, err)
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
