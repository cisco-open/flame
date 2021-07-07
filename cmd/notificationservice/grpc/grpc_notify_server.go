package grpcnotify

import (
	"net"
	"strconv"
	"time"

	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/types/known/anypb"

	pbNotification "wwwin-github.cisco.com/eti/fledge/pkg/proto/go/notification"
)

//notificationServer implement the notification service and include - proto unimplemented method and
//maintains list of connected clients & their streams.
type notificationServer struct {
	clients       map[string]*pbNotification.AgentInfo
	clientStreams map[string]*pbNotification.NotificationStreamingStore_SetupAgentStreamServer

	pbNotification.UnimplementedNotificationStreamingStoreServer
}

func (s *notificationServer) init() {
	s.clients = make(map[string]*pbNotification.AgentInfo)
	s.clientStreams = make(map[string]*pbNotification.NotificationStreamingStore_SetupAgentStreamServer)
}

//todo testing method to be removed
func testNotification(s *notificationServer) {
	zap.S().Debugf("Scheduling testing notification...")
	type Test struct {
		JobId string
	}
	time.AfterFunc(10*time.Second, func() {
		msg := &pbNotification.NewJobMessage{
			DesignId: "60d0cb55cba0514deadfb833",
			JobId:    "716af12b787d9ef0a",
		}
		m, err := anypb.New(msg)
		if err != nil {
			zap.S().Errorf("could not serialize message")
		}
		s.sendNotification("local", m)
	})
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
	pbNotification.RegisterNotificationStreamingStoreServer(s, server)

	//testing notification
	//todo tobe removed
	go testNotification(server)

	zap.S().Infof("Notification GRPC server listening at %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		zap.S().Errorf("failed to serve: %s", err)
	}
}
