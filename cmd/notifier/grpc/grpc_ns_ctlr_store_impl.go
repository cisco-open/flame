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
	"context"
	"errors"

	"go.uber.org/zap"

	"wwwin-github.cisco.com/eti/fledge/pkg/objects"
	pbNotification "wwwin-github.cisco.com/eti/fledge/pkg/proto/go/notification"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

func (s *notificationServer) JobNotification(ctx context.Context, in *pbNotification.JsonRequest) (*pbNotification.Response, error) {
	//decode request
	jobMsg := objects.JobNotification{}
	err := util.ProtoStructToStruct(in.GetDetails(), &jobMsg)
	if err != nil {
		zap.S().Errorf("error reading the request. %v", err)
		return nil, err
	}

	//notification handler
	var nsType pbNotification.StreamResponse_ResponseType
	switch jobMsg.NotificationType {
	case util.InitState:
		nsType = pbNotification.StreamResponse_JOB_NOTIFICATION_INIT
	case util.StartState:
		nsType = pbNotification.StreamResponse_JOB_NOTIFICATION_START
	case util.ReloadState:
		nsType = pbNotification.StreamResponse_JOB_NOTIFICATION_RELOAD
	default:
		zap.S().Errorf("invalid job notification type: %s", jobMsg.NotificationType)
		return nil, errors.New("invalid job notification type")
	}
	zap.S().Debugf("Sending job notification to the following clients %v. Notification type: %s", jobMsg.Agents, jobMsg.NotificationType)

	eList := map[string]interface{}{}
	for _, item := range jobMsg.Agents {
		cId := item.Uuid
		//clients = append(clients, item.Uuid)
		req := objects.AppConf{
			JobInfo:    jobMsg.Job,
			SchemaInfo: jobMsg.SchemaInfo,
			Role:       item.Role,
			Command:    item.Command,
		}
		err = s.pushNotification(cId, nsType, req)
		if err != nil {
			eList[cId] = err
		}
	}

	//notifications sent. However, check if it was sent to all or partial only.
	resp := &pbNotification.Response{Status: pbNotification.Response_SUCCESS}
	if len(eList) != 0 {
		zap.S().Errorf("error while sending out notification, only partial clients notified %v", eList)
		eList, _ := util.ToProtoStruct(eList)
		resp.Status = pbNotification.Response_SUCCESS_WITH_ERROR
		resp.Message = "error notifying all the agents"
		resp.Details = eList
	}
	return resp, nil
}
