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
	"net"
	"sync"
	"time"

	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"

	pbMeta "github.com/cisco-open/flame/pkg/proto/meta"
)

// meta server is a service to facilitate flame's data plane operations
type metaServer struct {
	mutex sync.Mutex
	jobs  map[string]*job

	pbMeta.UnimplementedMetaRouteServer
}

func Start(portNo uint16) {
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", portNo))
	if err != nil {
		zap.S().Errorf("failed to listen grpc server: %v", err)
	}

	// create grpc server
	s := grpc.NewServer()
	server := &metaServer{
		jobs: make(map[string]*job),
	}

	pbMeta.RegisterMetaRouteServer(s, server)
	reflection.Register(s)

	zap.S().Infof("MetaServer listening at %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		zap.S().Fatalf("failed to serve: %s", err)
	}
}

func (s *metaServer) RegisterMetaInfo(ctx context.Context, in *pbMeta.MetaInfo) (*pbMeta.MetaResponse, error) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	zap.S().Infof("jobId: %s, ChName: %s, Me: %s, Other: %s, Endpoint: %s", in.JobId, in.ChName, in.Me, in.Other, in.Endpoint)
	j, ok := s.jobs[in.JobId]
	if !ok {
		j = &job{channels: make(map[string]*channel)}
		s.jobs[in.JobId] = j
	}

	err := j.register(in)
	if err != nil {
		zap.S().Errorf("failed to register: %v", err)
		return nil, fmt.Errorf("failed to register: %v", err)
	}

	err = s.setupEndpointTimeout(in)
	if err != nil {
		zap.S().Errorf("failed to setup endpoint timeout: %v", err)
		return nil, err
	}

	resp := &pbMeta.MetaResponse{
		Status:    pbMeta.MetaResponse_SUCCESS,
		Endpoints: make([]string, 0),
	}
	endpoints := j.search(in.ChName, in.Other)
	for endpoint := range endpoints {
		resp.Endpoints = append(resp.Endpoints, endpoint)
	}

	return resp, nil
}

func (s *metaServer) HeartBeat(ctx context.Context, in *pbMeta.MetaInfo) (*pbMeta.MetaResponse, error) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	err := s.setupEndpointTimeout(in)
	if err != nil {
		return nil, err
	}

	resp := &pbMeta.MetaResponse{
		Status:    pbMeta.MetaResponse_SUCCESS,
		Endpoints: make([]string, 0),
	}
	return resp, nil
}

func (s *metaServer) setupEndpointTimeout(in *pbMeta.MetaInfo) error {
	j, ok := s.jobs[in.JobId]
	if !ok {
		return fmt.Errorf("job id %s not found", in.JobId)
	}

	endpoints := j.search(in.ChName, in.Me)
	if endpoints == nil {
		return fmt.Errorf("no endpoint found for  role %s", in.Me)
	}

	heartbeat, ok := endpoints[in.Endpoint]
	if !ok {
		return fmt.Errorf("no cancel channel found for endpoint %s", in.Endpoint)
	}

	if heartbeat != nil {
		heartbeat <- true
	} else {
		heartbeat = make(chan bool)
		endpoints[in.Endpoint] = heartbeat
		go s.cleanStaleEndpiont(endpoints, heartbeat, in.Endpoint)
	}

	return nil
}

func (s *metaServer) cleanStaleEndpiont(endpoints map[string]chan bool, heartbeat chan bool, endpoint string) {
	timer := time.NewTimer(TIMEOUT_STALE_ENDPOINT)

	for {
		select {
		case <-timer.C:
			zap.S().Infof("timer fired for endpoint %s", endpoint)

			s.mutex.Lock()
			delete(endpoints, endpoint)
			s.mutex.Unlock()
			return

		case <-heartbeat:
			// reset timer
			timer.Reset(TIMEOUT_STALE_ENDPOINT)
			zap.S().Infof("timer reset for endpoint %s", endpoint)
		}
	}
}
