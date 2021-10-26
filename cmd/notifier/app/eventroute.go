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
	"go.uber.org/zap"

	pbNotify "github.com/cisco/fledge/pkg/proto/notification"
)

// GetEvent is called by the client to subscribe to the notification service.
// Adds the client to the server client map and stores the client stream.
func (s *notificationServer) GetEvent(in *pbNotify.AgentInfo, stream pbNotify.EventRoute_GetEventServer) error {
	zap.S().Debugf("Serving event for agent %v", in)

	agentId := in.GetId()

	eventCh := s.getEventChannel(agentId)
	for {
		select {
		case event := <-eventCh:
			zap.S().Infof("Pushing event %v to agent %s", event, agentId)
			err := stream.Send(event)
			if err != nil {
				zap.S().Warnf("Failed to push notification to agent %s: %v", agentId, err)
			}

		case <-stream.Context().Done():
			zap.S().Infof("Stream context is done for agent %s", agentId)
			s.mutex.Lock()
			delete(s.eventQueues, agentId)
			s.mutex.Unlock()
			return nil
		}
	}
}

func (s *notificationServer) getEventChannel(agentId string) chan *pbNotify.Event {
	var eventCh chan *pbNotify.Event

	s.mutex.Lock()
	if _, ok := s.eventQueues[agentId]; !ok {
		eventCh = make(chan *pbNotify.Event)
		s.eventQueues[agentId] = eventCh
	} else {
		eventCh = s.eventQueues[agentId]
	}
	s.mutex.Unlock()

	return eventCh
}
