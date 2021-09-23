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
	pbNotify "wwwin-github.cisco.com/eti/fledge/pkg/proto/notification"
)

var ControllerGRPC = &controllerGRPC{}

//controllerGRPC implement the controller grpc server which is used by the REST API to send user requests
//also maintains connection with other services example: notification service.
type controllerGRPC struct {
	notificationServiceClient pbNotify.TriggerRouteClient
}

// InitGRPCService starts the controller grpc server and establishes connection with the notification service
func InitGRPCService(endpoint string) {
	ControllerGRPC.connectToNotificationService(endpoint)
}
