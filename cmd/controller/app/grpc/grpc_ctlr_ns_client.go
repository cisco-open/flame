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

package grpcctlr

import (
	"context"

	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/types/known/structpb"

	pbNotify "wwwin-github.cisco.com/eti/fledge/pkg/proto/notification"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

const (
	JobNotification = "JobNotification"
)

type fn func(context.Context, *pbNotify.EventRequest, ...grpc.CallOption) (*pbNotify.Response, error)

var notificationApiStore map[string]fn

// ConnectToNotificationService establishes connection to the notification service and stores the client object.
// The client object is later used by the controller to pass information to the notification service which passes it to the fledgelet.
func (s *controllerGRPC) connectToNotificationService(endpoint string) {
	conn, err := grpc.Dial(endpoint, grpc.WithInsecure())
	if err != nil {
		zap.S().Fatalf("cannot connect to notification service %v", err)
	}

	//grpc client for future reuse
	s.notificationServiceClient = pbNotify.NewTriggerRouteClient(conn)

	//adding grpc end points for generic use through SendNotification method
	notificationApiStore = map[string]fn{
		JobNotification: s.notificationServiceClient.Notify,
	}

	zap.S().Infof("Controller -- Notification service connection established. Notification service at %s", endpoint)
}

//SendNotification implements the generic method to call notification service end points.
func (s *controllerGRPC) SendNotification(endPoint string, in interface{}) (*pbNotify.Response, error) {
	//TODO - use ToProtoStruct
	//Step 1 - create notification object
	m, err := util.StructToMapInterface(in)
	if err != nil {
		zap.S().Errorf("error converting notification object into map interface. %v", err)
		return nil, err
	}
	_, err = structpb.NewStruct(m)
	if err != nil {
		zap.S().Errorf("error creating proto struct. %v", err)
		return nil, err
	}

	// FIXME: dummy event request
	req := &pbNotify.EventRequest{}

	//Step 2 - send grpc message
	response, err := notificationApiStore[endPoint](context.Background(), req)

	//Step 3 - handle response
	if err != nil {
		zap.S().Errorf("error sending out nofitification. Endpoint: %s", endPoint)
	}
	return response, err
}
