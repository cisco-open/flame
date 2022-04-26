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
	"go.uber.org/zap"

	pbNotify "github.com/cisco-open/flame/pkg/proto/notification"
)

const (
	eventChannelLen = 1
)

// GetEvent is called by the client to subscribe to the notification service.
// Adds the client to the server client map and stores the client stream.
func (s *notificationServer) GetEvent(in *pbNotify.TaskInfo, stream pbNotify.EventRoute_GetEventServer) error {
	zap.S().Debugf("Serving event for task %v", in)

	taskId := in.GetId()

	eventCh := s.getEventChannel(taskId)
	for {
		select {
		case event := <-eventCh:
			zap.S().Infof("Pushing event %v to task %s", event, taskId)
			err := stream.Send(event)
			if err != nil {
				zap.S().Warnf("Failed to push notification to task %s: %v", taskId, err)
			}

		case <-stream.Context().Done():
			zap.S().Infof("Stream context is done for task %s", taskId)
			s.mutex.Lock()
			delete(s.eventQueues, taskId)
			s.mutex.Unlock()
			return nil
		}
	}
}

func (s *notificationServer) getEventChannel(taskId string) chan *pbNotify.Event {
	var eventCh chan *pbNotify.Event

	s.mutex.Lock()
	if _, ok := s.eventQueues[taskId]; !ok {
		eventCh = make(chan *pbNotify.Event, eventChannelLen)
		s.eventQueues[taskId] = eventCh
	} else {
		eventCh = s.eventQueues[taskId]
	}
	s.mutex.Unlock()

	return eventCh
}
