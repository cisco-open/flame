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
	"fmt"

	"go.uber.org/zap"

	pbNotify "github.com/cisco/fledge/pkg/proto/notification"
)

// GetEvent is called by the client to subscribe to the notification service.
// Adds the client to the server client map and stores the client stream.
func (s *notificationServer) GetEvent(in *pbNotify.AgentInfo, stream pbNotify.EventRoute_GetEventServer) error {
	s.addNewClient(in, &stream)

	// the stream should not be killed so we do not return from this server
	// loop infinitely to keep stream alive else this stream will be closed
	// TODO: refactor this by using channel
	select {}
}

// addNewClient is responsible to add new client to the server map.
func (s *notificationServer) addNewClient(in *pbNotify.AgentInfo, stream *pbNotify.EventRoute_GetEventServer) {
	zap.S().Debugf("Adding new agent to the collection | %v", in)

	agentId := in.GetId()

	s.clientStreams[agentId] = stream
	s.clients[agentId] = in
}

// pushNotification sends a notification message to a specific agent
func (s *notificationServer) pushNotification(agentId string, event *pbNotify.Event) error {
	zap.S().Debugf("Sending notification to client: %v", agentId)

	stream := s.clientStreams[agentId]
	if stream == nil {
		errMsg := fmt.Sprintf("agent %s is unregistered", agentId)
		zap.S().Warn(errMsg)
		return fmt.Errorf(errMsg)
	}

	err := (*stream).Send(event)
	if err != nil {
		errMsg := fmt.Sprintf("failed to push notification to agent %s: %v", agentId, err)
		zap.S().Warn(errMsg)
		return fmt.Errorf(errMsg)
	}

	return nil
}
