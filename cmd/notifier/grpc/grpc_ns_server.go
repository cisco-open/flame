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

package grpcnotify

import (
	"net"
	"strconv"

	"go.uber.org/zap"
	"google.golang.org/grpc"

	pbNotification "wwwin-github.cisco.com/eti/fledge/pkg/proto/go/notification"
)

//notificationServer implement the notification service and include - proto unimplemented method and
//maintains list of connected clients & their streams.
type notificationServer struct {
	clients       map[string]*pbNotification.AgentInfo
	clientStreams map[string]*pbNotification.NotificationStreamingStore_SetupAgentStreamServer

	pbNotification.UnimplementedNotificationStreamingStoreServer
	pbNotification.UnimplementedNotificationControllerStoreServer
}

func (s *notificationServer) init() {
	s.clients = make(map[string]*pbNotification.AgentInfo)
	s.clientStreams = make(map[string]*pbNotification.NotificationStreamingStore_SetupAgentStreamServer)
}

//StartGRPCService starts the notification grpc server and register the corresponding stores implemented by notificationServer.
func StartGRPCService(portNo int) {
	lis, err := net.Listen("tcp", ":"+strconv.Itoa(portNo))
	if err != nil {
		zap.S().Errorf("failed to listen grpc server: %v", err)
	}

	// create grpc server
	s := grpc.NewServer()
	server := &notificationServer{}
	server.init()

	//register grpc services
	pbNotification.RegisterNotificationStreamingStoreServer(s, server)
	pbNotification.RegisterNotificationControllerStoreServer(s, server)

	zap.S().Infof("Notification GRPC server listening at %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		zap.S().Errorf("failed to serve: %s", err)
	}
}
