// Copyright 2023 Cisco Systems, Inc. and its affiliates
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
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	"go.uber.org/zap"
	v1 "k8s.io/api/core/v1"

	"github.com/cisco-open/flame/pkg/openapi"
	"github.com/cisco-open/flame/pkg/openapi/constants"
	"github.com/cisco-open/flame/pkg/restapi"
)

const taskLimit = 10

func (r *resourceHandler) monitorJobs() {
	go r.monitorRunningJobs()
	go r.monitorPods()
}

func (r *resourceHandler) monitorRunningJobs() error {
	for {
		jobs, err := r.getJobs()
		if err != nil {
			zap.S().Errorf("failed to get jobs: %v", err)
			continue
		}

		for _, job := range jobs {
			if job.State == openapi.RUNNING {
				taskInfo, err := r.getTasks(job.Id, taskLimit)
				if err != nil {
					zap.S().Errorf("failed to get tasks: %v", err)
					continue
				}
				for _, task := range taskInfo {
					if task.State != openapi.RUNNING {
						continue
					}

					r.dplyr.MonitorTask(task.JobId, task.TaskId)
				}
			}
		}

		// wait for some time before checking the status of pods again
		time.Sleep(time.Minute)
	}
}

func (r *resourceHandler) getJobs() ([]openapi.JobStatus, error) {
	// construct URL
	uriMap := map[string]string{
		constants.ParamComputeID: r.spec.ComputeId,
	}
	url := restapi.CreateURL(r.apiserverEp, restapi.GetJobsByComputeEndPoint, uriMap)

	code, body, err := restapi.HTTPGet(url)
	if err != nil || restapi.CheckStatusCode(code) != nil {
		var msg string
		_ = json.Unmarshal(body, &msg)
		zap.S().Errorf("Failed to retrieve jobs' status - code: %d; %s\n", code, msg)
		return nil, nil
	}

	// convert the response into list of struct
	var infoList []openapi.JobStatus
	err = json.Unmarshal(body, &infoList)
	if err != nil {
		zap.S().Errorf("Failed to unmarshal job status: %v\n", err)
		return nil, err
	}

	return infoList, nil
}

func (r *resourceHandler) monitorPods() {
	if err := r.dplyr.Initialize("", r.namespace); err != nil {
		zap.S().Errorf("failed to initialize a job deployer: %v", err)
		return
	}

	for {
		taskHealthDetails, err := r.dplyr.GetMonitoredPodStatuses()
		if err != nil {
			zap.S().Errorf("failed to check pods: %v", err)
		} else if len(taskHealthDetails) > 0 {
			for _, pod := range taskHealthDetails {
				if pod.Status.Phase == v1.PodPending ||
					pod.Status.Phase == v1.PodRunning {
					continue
				}

				now := time.Now().UTC()

				err := r.updateTaskStatus(pod.JobID, pod.TaskID, openapi.TaskStatus{
					State:     openapi.FAILED,
					Timestamp: now,
				})
				if err != nil {
					zap.S().Errorf("failed to update task status: %v", err)
					continue
				}

				err = r.updateJobStatus(pod.JobID, openapi.JobStatus{
					Id:    pod.JobID,
					State: openapi.FAILED,
				})
				if err != nil {
					zap.S().Errorf("failed to update job status: %v", err)
					continue
				}

				r.dplyr.DeleteTaskFromMonitoring(pod.TaskID)
			}
		}

		// wait for some time before checking the status of pods again
		time.Sleep(time.Minute)
	}
}

func (r *resourceHandler) updateTaskStatus(jobId, taskId string, taskStatus openapi.TaskStatus) error {
	zap.S().Debugf("Updating task status for job %s | task: %s", jobId, taskId)

	// create controller request
	uriMap := map[string]string{
		constants.ParamJobID:  jobId,
		constants.ParamTaskID: taskId,
	}
	url := restapi.CreateURL(r.apiserverEp, restapi.UpdateTaskStatusEndPoint, uriMap)

	// send put request
	code, _, err := restapi.HTTPPut(url, taskStatus, "application/json")
	if err != nil || restapi.CheckStatusCode(code) != nil {
		zap.S().Errorf("Failed to update task status %v for job %s - code: %d, error: %v\n",
			taskStatus, jobId, code, err)
		return err
	}

	return nil
}

func (r *resourceHandler) updateJobStatus(jobId string, status openapi.JobStatus) error {
	zap.S().Debugf("Updating task status for job %s", jobId)

	// create controller request
	uriMap := map[string]string{
		constants.ParamJobID: jobId,
	}
	url := restapi.CreateURL(r.apiserverEp, restapi.UpdateJobStatusEndPoint, uriMap)

	// send put request
	code, _, err := restapi.HTTPPut(url, status, "application/json")
	if err != nil || restapi.CheckStatusCode(code) != nil {
		zap.S().Errorf("Failed to update job status %v for job %s - code: %d, error: %v\n",
			status, jobId, code, err)
		return err
	}

	return nil
}

func (r *resourceHandler) getTasks(jobId string, limit int) ([]openapi.TaskInfo, error) {
	uriMap := map[string]string{
		constants.ParamUser:    r.spec.ComputeId,
		constants.ParamJobID:   jobId,
		constants.ParamLimit:   strconv.Itoa(limit),
		constants.ParamGeneric: "true",
	}
	url := restapi.CreateURL(r.apiserverEp, restapi.GetTasksInfoEndpoint, uriMap)

	code, body, err := restapi.HTTPGet(url)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch task")
	}

	if err = restapi.CheckStatusCode(code); err != nil {
		return nil, err
	}

	// convert the response into list of struct
	var taskInfo []openapi.TaskInfo
	err = json.Unmarshal(body, &taskInfo)
	if err != nil {
		fmt.Printf("Failed to unmarshal job status: %v\n", err)
		return nil, err
	}

	return taskInfo, nil
}
