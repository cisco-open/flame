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
	"fmt"
	"net"
	"os"
	"strconv"

	"go.uber.org/zap"
	"google.golang.org/grpc"

	pbAgent "github.com/cisco-open/flame/pkg/proto/agent"
	"github.com/cisco-open/flame/pkg/util"
)

const (
	envTaskId  = "FLAME_TASK_ID"
	envTaskKey = "FLAME_TASK_KEY"
)

type AgentService struct {
	apiserverEp string
	notifierEp  string
	name        string
	id          string
	tHandler    *taskHandler
}

func NewAgent(apiserverEp string, notifierEp string) (*AgentService, error) {
	name, err := os.Hostname()
	if err != nil {
		return nil, err
	}

	taskId := os.Getenv(envTaskId)
	// in case agent id env variable is not set
	if taskId == "" {
		return nil, fmt.Errorf("task id not found from env variable %s", envTaskId)
	}

	taskKey := os.Getenv(envTaskKey)
	if taskKey == "" {
		return nil, fmt.Errorf("agent key not found from env variable %s", envTaskKey)
	}

	tHandler := newTaskHandler(apiserverEp, notifierEp, name, taskId, taskKey)

	agent := &AgentService{
		apiserverEp: apiserverEp,
		notifierEp:  notifierEp,
		name:        name,
		id:          taskId,
		tHandler:    tHandler,
	}

	return agent, nil
}

func (agent *AgentService) Start() error {
	zap.S().Infof("Starting %s... name: %s | id: %s", util.Agent, agent.name, agent.id)

	agent.tHandler.start()

	err := agent.startAppServer()

	return err
}

// startAppServer starts the flamelet grpc server and register the corresponding stores implemented by agentServer.
func (agent *AgentService) startAppServer() error {
	lis, err := net.Listen("tcp", ":"+strconv.Itoa(util.AgentGrpcPort))
	if err != nil {
		zap.S().Errorf("Failed to listen grpc server: %v", err)
		return err
	}

	// create grpc server
	s := grpc.NewServer()
	server := &appServer{}
	server.init()

	//register grpc services
	pbAgent.RegisterStreamingStoreServer(s, server)

	zap.S().Infof("Agent GRPC server listening at %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		zap.S().Errorf("Failed to serve: %s", err)
		return err
	}

	return nil
}
