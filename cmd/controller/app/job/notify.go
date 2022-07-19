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

package job

import (
	"context"
	"crypto/tls"
	"fmt"

	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"

	pbNotify "github.com/cisco-open/flame/pkg/proto/notification"
)

// notifyClient implements a notification client that sends a notification request
// to a notifier.
type notifyClient struct {
	endpoint string

	grpcDialOpt grpc.DialOption
}

func newNotifyClient(endpoint string, bInsecure bool, bPlain bool) *notifyClient {
	var grpcDialOpt grpc.DialOption

	if bPlain {
		grpcDialOpt = grpc.WithTransportCredentials(insecure.NewCredentials())
	} else {
		tlsCfg := &tls.Config{}
		if bInsecure {
			zap.S().Warn("Warning: allow insecure connection\n")

			tlsCfg.InsecureSkipVerify = true
		}
		grpcDialOpt = grpc.WithTransportCredentials(credentials.NewTLS(tlsCfg))
	}

	return &notifyClient{endpoint: endpoint, grpcDialOpt: grpcDialOpt}
}

// sendNotification sends a notification request to the notifier
func (nc *notifyClient) sendNotification(req *pbNotify.JobEventRequest) (*pbNotify.JobResponse, error) {
	conn, err := grpc.Dial(nc.endpoint, nc.grpcDialOpt)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to notifier: %v", err)
	}
	defer conn.Close()

	trClient := pbNotify.NewJobTriggerRouteClient(conn)

	zap.S().Infof("Successfully connected to notifier: %s", nc.endpoint)

	response, err := trClient.NotifyJob(context.Background(), req)

	if err != nil {
		errMsg := fmt.Sprintf("notification failed: %v", err)
		zap.S().Warn(errMsg)
		return nil, fmt.Errorf(errMsg)
	}

	return response, nil
}
