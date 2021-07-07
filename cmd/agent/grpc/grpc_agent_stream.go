package grpcagent

import (
	"context"
	"io"
	"log"
	"strconv"

	"go.uber.org/zap"
	"google.golang.org/grpc"

	pbNotification "wwwin-github.cisco.com/eti/fledge/pkg/proto/go/notification"
)

//ConnectToNotificationService connects to the notification grpc server.
//It starts a new goroutine which listens for notifications.
func ConnectToNotificationService(ip string, port int) {
	//dial server
	conn, err := grpc.Dial(ip+":"+strconv.Itoa(port), grpc.WithInsecure())
	if err != nil {
		zap.S().Fatalf("can not connect with server %v", err)
	}

	client := pbNotification.NewNotificationStreamingStoreClient(conn)
	//todo name and ip to be fetched using api
	in := &pbNotification.AgentInfo{
		Uuid: "local",
		Ip:   "localhost",
	}

	//setup notification stream
	stream, err := client.SetupAgentStream(context.Background(), in)
	if err != nil {
		zap.S().Fatalf("open stream error %v", err)
	}

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
				log.Fatalf("cannot receive %v", err)
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
	case pbNotification.StreamResponse_JOB_NOTIFICATION:
		msg := &pbNotification.NewJobMessage{}
		in.GetMessage().UnmarshalTo(msg)
		newJob(msg)
		break
	default:
		zap.S().Errorf("Invalid message type: %s", in.GetType())
	}
}

func newJob(in *pbNotification.NewJobMessage) {
	zap.S().Infof("New job request submitted. %v", in)
}
