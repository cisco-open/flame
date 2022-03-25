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
	"net"
	"sync"

	"go.uber.org/zap"
	"google.golang.org/grpc"

	pbNotify "github.com/cisco-open/flame/pkg/proto/notification"
)

// notificationServer implement the notification service and include - proto unimplemented method and
// maintains list of connected clients & their streams.
type notificationServer struct {
	eventQueues map[string]chan *pbNotify.Event
	mutex       sync.Mutex

	pbNotify.UnimplementedEventRouteServer
	pbNotify.UnimplementedTriggerRouteServer
}

// StartGRPCService starts the notification grpc server
func StartGRPCService(portNo uint16) {
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", portNo))
	if err != nil {
		zap.S().Errorf("failed to listen grpc server: %v", err)
	}

	// create grpc server
	s := grpc.NewServer()
	server := &notificationServer{
		eventQueues: make(map[string]chan *pbNotify.Event),
	}

	// register grpc services
	pbNotify.RegisterEventRouteServer(s, server)
	pbNotify.RegisterTriggerRouteServer(s, server)

	zap.S().Infof("Notification GRPC server listening at %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		zap.S().Errorf("failed to serve: %s", err)
	}
}
