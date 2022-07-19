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
		return nil, fmt.Errorf("unknown event type: %s", in.GetType())
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
