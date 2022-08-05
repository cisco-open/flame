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
	"fmt"

	"go.uber.org/zap"

	pbNotify "github.com/cisco-open/flame/pkg/proto/notification"
)

func (s *notificationServer) NotifyJob(ctx context.Context, in *pbNotify.JobEventRequest) (*pbNotify.JobResponse, error) {
	switch in.Type {
	case pbNotify.JobEventType_START_JOB:
	case pbNotify.JobEventType_STOP_JOB:
	case pbNotify.JobEventType_UPDATE_JOB:

	case pbNotify.JobEventType_UNKNOWN_EVENT_TYPE:
		fallthrough
	default:
		return nil, fmt.Errorf("unknown job event type: %s", in.GetType())
	}

	failedTasks := make([]string, 0)
	for _, taskId := range in.TaskIds {
		event := pbNotify.JobEvent{
			Type:  in.Type,
			JobId: in.JobId,
		}

		eventCh := s.getJobEventChannel(taskId)

		select {
		case eventCh <- &event:
			// Do nothing
		default:
			failedTasks = append(failedTasks, taskId)
		}
	}

	resp := &pbNotify.JobResponse{
		Status:      pbNotify.JobResponse_SUCCESS,
		Message:     "Successfully registered event for all tasks",
		FailedTasks: failedTasks,
	}

	if len(in.TaskIds) > 0 && len(failedTasks) == len(in.TaskIds) {
		resp.Message = "Failed to register event for all tasks"
		resp.Status = pbNotify.JobResponse_ERROR
	} else if len(failedTasks) > 0 && len(failedTasks) < len(in.TaskIds) {
		resp.Message = "Registered event for some tasks successfully"
		resp.Status = pbNotify.JobResponse_PARTIAL_SUCCESS
	}

	zap.S().Info(resp.Message)

	return resp, nil
}

func (s *notificationServer) NotifyDeploy(ctx context.Context, in *pbNotify.DeployEventRequest) (*pbNotify.DeployResponse, error) {
	zap.S().Info("TriggerRoute - received message from controller to %v for compute %v", in.GetType(), in.ComputeIds)
	switch in.Type {
	case pbNotify.DeployEventType_ADD_RESOURCE:
	case pbNotify.DeployEventType_REVOKE_RESOURCE:

	case pbNotify.DeployEventType_UNKNOWN_DEPLOYMENT_TYPE:
		fallthrough
	default:
		return nil, fmt.Errorf("unknown deploy event type: %s", in.GetType())
	}

	failedDeployers := make([]string, 0)
	for _, computeId := range in.ComputeIds {
		zap.S().Infof("Going to send deploy event to deployer %s", computeId)
		event := pbNotify.DeployEvent{
			Type:  in.Type,
			JobId: in.JobId,
		}

		eventCh := s.getDeployEventChannel(computeId)

		select {
		case eventCh <- &event:
			// Do nothing
		default:
			failedDeployers = append(failedDeployers, computeId)
			zap.S().Infof("Failed to send deploy event to deployer %s, updated failedDeployers: %v", computeId, failedDeployers)
		}
	}

	resp := &pbNotify.DeployResponse{
		Status:          pbNotify.DeployResponse_SUCCESS,
		Message:         "Successfully issued deployment instructions to deployers",
		FailedDeployers: failedDeployers,
	}

	if len(in.ComputeIds) > 0 && len(failedDeployers) == len(in.ComputeIds) {
		resp.Message = "Failed to issue deployment instructions for all deployers"
		resp.Status = pbNotify.DeployResponse_ERROR
	} else if len(failedDeployers) > 0 && len(failedDeployers) < len(in.ComputeIds) {
		resp.Message = "Issued deployment instructions for some deployers successfully"
		resp.Status = pbNotify.DeployResponse_PARTIAL_SUCCESS
	}

	zap.S().Info(resp.Message)

	return resp, nil
}
