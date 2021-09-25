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
	"time"

	backoff "github.com/cenkalti/backoff/v4"
	"go.uber.org/zap"
	"google.golang.org/grpc"

	pbNotify "wwwin-github.cisco.com/eti/fledge/pkg/proto/notification"
)

type NotifyHandler struct {
	apiserverEp string
	notifierEp  string
	name        string
	agentId     string

	stream pbNotify.EventRoute_GetEventClient
}

func newNotifyHandler(apiserverEp string, notifierEp string, name string, agentId string) *NotifyHandler {
	return &NotifyHandler{
		apiserverEp: apiserverEp,
		notifierEp:  notifierEp,
		name:        name,
		agentId:     agentId,
	}
}

// start connects to the notifier via grpc and handles notifications from the notifier
func (h *NotifyHandler) start() {
	go h.doStart()
}

func (h *NotifyHandler) doStart() {
	for {
		expBackoff := backoff.NewExponentialBackOff()
		expBackoff.MaxElapsedTime = 5 * time.Minute // max wait time: 5 minutes
		err := backoff.Retry(h.connect, expBackoff)
		if err != nil {
			zap.S().Fatalf("Cannot connect with notifier: %v", err)
		}

		h.do()
	}
}

func (h *NotifyHandler) connect() error {
	// dial server
	conn, err := grpc.Dial(h.notifierEp, grpc.WithInsecure())
	if err != nil {
		zap.S().Debugf("Cannot connect with notifier: %v", err)
		return err
	}

	client := pbNotify.NewEventRouteClient(conn)
	in := &pbNotify.AgentInfo{
		Id:       h.agentId,
		Hostname: h.name,
	}

	// setup notification stream
	stream, err := client.GetEvent(context.Background(), in)
	if err != nil {
		zap.S().Debugf("Open stream error: %v", err)
		return err
	}

	h.stream = stream
	zap.S().Infof("Connected with notifier at %s", h.notifierEp)

	return nil
}

func (h *NotifyHandler) do() {
	for {
		resp, err := h.stream.Recv()
		if err != nil {
			zap.S().Errorf("Failed to receive notification: %v", err)
			break
		}

		h.dealWith(resp)
	}

	zap.S().Info("Disconnected from notifier")
}

//newNotification acts as a handler and calls respective functions based on the response type to act on the received notifications.
func (h *NotifyHandler) dealWith(in *pbNotify.Event) {
	switch in.GetType() {
	case pbNotify.EventType_START_JOB:
		h.startJob(in.JobId)

	case pbNotify.EventType_STOP_JOB:
		h.stopJob(in.JobId)

	case pbNotify.EventType_UPDATE_JOB:
		h.updateJob(in.JobId)

	case pbNotify.EventType_UNKNOWN_EVENT_TYPE:
		fallthrough
	default:
		zap.S().Errorf("Invalid message type: %s", in.GetType())
	}
}
