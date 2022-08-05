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

// GetJobEvent is called by the client to subscribe to the notification service.
// Adds the client to the server client map and stores the client stream.
func (s *notificationServer) GetJobEvent(in *pbNotify.JobTaskInfo, stream pbNotify.JobEventRoute_GetJobEventServer) error {
	zap.S().Debugf("Serving event for task %v", in)

	taskId := in.GetId()

	eventCh := s.getJobEventChannel(taskId)
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
			s.mutexJob.Lock()
			delete(s.jobEventQueues, taskId)
			s.mutexJob.Unlock()
			return nil
		}
	}
}

func (s *notificationServer) getJobEventChannel(taskId string) chan *pbNotify.JobEvent {
	var eventCh chan *pbNotify.JobEvent

	s.mutexJob.Lock()
	if _, ok := s.jobEventQueues[taskId]; !ok {
		eventCh = make(chan *pbNotify.JobEvent, eventChannelLen)
		s.jobEventQueues[taskId] = eventCh
	} else {
		eventCh = s.jobEventQueues[taskId]
	}
	s.mutexJob.Unlock()

	return eventCh
}

// GetDeployEvent is called by the client to subscribe to the notification service.
// Adds the client to the server client map and stores the client stream.
func (s *notificationServer) GetDeployEvent(in *pbNotify.DeployInfo, stream pbNotify.DeployEventRoute_GetDeployEventServer) error {
	zap.S().Debugf("Serving event for deployer %v", in)

	computeId := in.GetComputeId()

	eventCh := s.getDeployEventChannel(computeId)
	for {
		select {
		case event := <-eventCh:
			zap.S().Infof("Pushing event %v to deployer %s", event, computeId)
			err := stream.Send(event)
			if err != nil {
				zap.S().Warnf("Failed to push notification to deployer %s: %v", computeId, err)
			}

		case <-stream.Context().Done():
			zap.S().Infof("Stream context is done for deployer %s", computeId)
			s.mutexDeploy.Lock()
			delete(s.deployEventQueues, computeId)
			s.mutexDeploy.Unlock()
			return nil
		}
	}
}

func (s *notificationServer) getDeployEventChannel(computeId string) chan *pbNotify.DeployEvent {
	var eventCh chan *pbNotify.DeployEvent
	zap.S().Infof("Getting deploy event channel for deployer %s", computeId)

	s.mutexDeploy.Lock()
	if _, ok := s.deployEventQueues[computeId]; !ok {
		eventCh = make(chan *pbNotify.DeployEvent, eventChannelLen)
		s.deployEventQueues[computeId] = eventCh
		zap.S().Infof("Couldn't get existing channel. Created deploy event channel for deployer %s", computeId)
	} else {
		eventCh = s.deployEventQueues[computeId]
	}
	s.mutexDeploy.Unlock()

	return eventCh
}
