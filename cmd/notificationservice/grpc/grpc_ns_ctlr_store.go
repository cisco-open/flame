package grpcnotify

import (
	"context"

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
	var clients []string
	for _, item := range jobMsg.Agents {
		clients = append(clients, item.Uuid)
	}

	var nsType pbNotification.StreamResponse_ResponseType
	if nsType = pbNotification.StreamResponse_JOB_NOTIFICATION_INIT; jobMsg.NotificationType == util.Start {
		nsType = pbNotification.StreamResponse_JOB_NOTIFICATION_START
	}
	zap.S().Debugf("Sending job notification to the following clients %v. Notification type: %s", clients, jobMsg.NotificationType)

	errList := s.pushNotifications(clients, nsType, jobMsg.Job)
	//notifications sent. However, check if it was sent to all or partial only.
	resp := &pbNotification.Response{Status: pbNotification.Response_SUCCESS}
	if errList != nil {
		zap.S().Errorf("error while sending out notification, only partial clients notified %v", errList)
		eList, _ := util.ToProtoStruct(errList)
		resp.Status = pbNotification.Response_SUCCESS_WITH_ERROR
		resp.Message = "error notifying all the agents"
		resp.Details = eList
	}
	return resp, nil
}
