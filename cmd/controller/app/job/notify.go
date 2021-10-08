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

package job

import (
	"context"
	"fmt"

	"go.uber.org/zap"
	"google.golang.org/grpc"

	pbNotify "github.com/cisco/fledge/pkg/proto/notification"
)

// notifyClient implements a notification client that sends a notification request
// to a notifier.
type notifyClient struct {
	endpoint string
}

func newNotifyClient(endpoint string) *notifyClient {
	return &notifyClient{endpoint: endpoint}
}

// sendNotification sends a notification request to the notifier
func (nc *notifyClient) sendNotification(req *pbNotify.EventRequest) (*pbNotify.Response, error) {
	conn, err := grpc.Dial(nc.endpoint, grpc.WithInsecure())
	if err != nil {
		return nil, fmt.Errorf("failed to connect to notifier: %v", err)
	}
	defer conn.Close()

	trClient := pbNotify.NewTriggerRouteClient(conn)

	zap.S().Infof("Successfully connected to notifier: %s", nc.endpoint)

	response, err := trClient.Notify(context.Background(), req)

	if err != nil {
		errMsg := fmt.Sprintf("notification failed: %v", err)
		zap.S().Warn(errMsg)
		return nil, fmt.Errorf(errMsg)
	}

	return response, nil
}
