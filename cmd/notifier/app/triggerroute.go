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
	"fmt"

	"go.uber.org/zap"

	pbNotify "github.com/cisco-open/flame/pkg/proto/notification"
)

func (s *notificationServer) Notify(ctx context.Context, in *pbNotify.EventRequest) (*pbNotify.Response, error) {
	switch in.Type {
	case pbNotify.EventType_START_JOB:
	case pbNotify.EventType_STOP_JOB:
	case pbNotify.EventType_UPDATE_JOB:

	case pbNotify.EventType_UNKNOWN_EVENT_TYPE:
		fallthrough
	default:
		return nil, fmt.Errorf("unknown event type: %s", in.GetType())
	}

	failedAgents := make([]string, 0)
	for _, agentId := range in.AgentIds {
		event := pbNotify.Event{
			Type:  in.Type,
			JobId: in.JobId,
		}

		eventCh := s.getEventChannel(agentId)

		select {
		case eventCh <- &event:
			// Do nothing
		default:
			failedAgents = append(failedAgents, agentId)
		}
	}

	resp := &pbNotify.Response{
		Status:       pbNotify.Response_SUCCESS,
		Message:      "Successfully registered event for all agents",
		FailedAgents: failedAgents,
	}

	if len(in.AgentIds) > 0 && len(failedAgents) == len(in.AgentIds) {
		resp.Message = "Failed to register event for all agents"
		resp.Status = pbNotify.Response_ERROR
	} else if len(failedAgents) > 0 && len(failedAgents) < len(in.AgentIds) {
		resp.Message = "Registered event for some agents successfully"
		resp.Status = pbNotify.Response_PARTIAL_SUCCESS
	}

	zap.S().Info(resp.Message)

	return resp, nil
}
