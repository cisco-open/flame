package app

import (
	"context"
	"io"

	"go.uber.org/zap"
	"google.golang.org/grpc"
	"wwwin-github.cisco.com/eti/fledge/pkg/objects"
	pbNotification "wwwin-github.cisco.com/eti/fledge/pkg/proto/go/notification"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

//ConnectToNotificationService connects to the notification grpc server.
//It starts a new goroutine which listens for notifications.
func ConnectToNotificationService() {
	//dial server
	conn, err := grpc.Dial(Agent.nsInfo.GetAddress(), grpc.WithInsecure())
	if err != nil {
		zap.S().Fatalf("can not connect with notification service %v", err)
	}

	client := pbNotification.NewNotificationStreamingStoreClient(conn)
	in := &pbNotification.AgentInfo{
		Uuid: Agent.uuid,
		Name: Agent.name,
	}

	//setup notification stream
	stream, err := client.SetupAgentStream(context.Background(), in)
	if err != nil {
		zap.S().Fatalf("open stream error %v", err)
	}
	zap.S().Infof("Agent -- Notification service connection established. Notification service at %v", Agent.nsInfo)

	//creating a channel to inform the client if notification connection is broken
	done := make(chan bool)

	//goroutine to wait and read for push notifications
	go func() {
		for {
			resp, err := stream.Recv()
			if err == io.EOF {
				done <- true //means stream is finished
				return
			} else if err != nil {
				zap.S().Errorf("cannot receive notification %v", err)
				done <- true //means stream is finished
			}
			newNotification(resp)
		}
	}()

	//todo implement re-connect functionality
	<-done
	zap.S().Errorf("notification service connection no longer active.")
}

//newNotification acts as a handler and calls respective functions based on the response type to act on the received notifications.
func newNotification(in *pbNotification.StreamResponse) {
	switch in.GetType() {
	case pbNotification.StreamResponse_JOB_NOTIFICATION_INIT:
		jobMsg := objects.AppConf{}
		err := util.ProtoStructToStruct(in.GetMessage(), &jobMsg)
		if err != nil {
			zap.S().Errorf("error processing the job request. %v", err)
		} else {
			NewJobInit(jobMsg)
		}
	case pbNotification.StreamResponse_JOB_NOTIFICATION_START:
		jobMsg := objects.AppConf{}
		err := util.ProtoStructToStruct(in.GetMessage(), &jobMsg)
		if err != nil {
			zap.S().Errorf("error processing the job request. %v", err)
		} else {
			NewJobStart(jobMsg)
		}
	default:
		zap.S().Errorf("Invalid message type: %s", in.GetType())
	}
}
