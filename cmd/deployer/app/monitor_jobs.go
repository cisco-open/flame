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
	"time"

	"go.uber.org/zap"
	v1 "k8s.io/api/core/v1"

	"github.com/cisco-open/flame/pkg/openapi"
	"github.com/cisco-open/flame/pkg/openapi/constants"
	"github.com/cisco-open/flame/pkg/restapi"
)

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

				r.dplyr.DeleteTaskFromMonitoring(pod.TaskID)
				zap.S().Info("task %s failed; pod status: %s", pod.TaskID, pod.Status.Phase)
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
