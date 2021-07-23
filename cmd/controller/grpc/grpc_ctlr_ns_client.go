package grpcctlr

import (
	"context"

	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/types/known/structpb"
	"wwwin-github.cisco.com/eti/fledge/pkg/objects"
	pbNotification "wwwin-github.cisco.com/eti/fledge/pkg/proto/go/notification"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

const (
	JobNotification = "JobNotification"
)

type fn func(context.Context, *pbNotification.JsonRequest, ...grpc.CallOption) (*pbNotification.Response, error)

var notificationApiStore map[string]fn

// ConnectToNotificationService establishes connection to the notification service and stores the client object.
// The client object is later used by the controller to pass information to the notification service which passes it to the agent.
func (s *controllerGRPC) connectToNotificationService(sInfo objects.ServerInfo) {
	conn, err := grpc.Dial(sInfo.GetAddress(), grpc.WithInsecure())
	if err != nil {
		zap.S().Fatalf("cannot connect to notification service %v", err)
	}

	//grpc client for future reuse
	s.notificationServiceClient = pbNotification.NewNotificationControllerStoreClient(conn)

	//adding grpc end points for generic use through SendNotification method
	notificationApiStore = map[string]fn{
		JobNotification: s.notificationServiceClient.JobNotification,
	}

	zap.S().Infof("Controller -- Notification service connection established. Notification service at %v", sInfo)
}

//SendNotification implements the generic method to call notification service end points.
func (s *controllerGRPC) SendNotification(endPoint string, in interface{}) (*pbNotification.Response, error) {
	//TODO - use ToProtoStruct
	//Step 1 - create notification object
	m, err := util.StructToMapInterface(in)
	if err != nil {
		zap.S().Errorf("error converting notification object into map interface. %v", err)
		return nil, err
	}
	details, err := structpb.NewStruct(m)
	if err != nil {
		zap.S().Errorf("error creating proto struct. %v", err)
		return nil, err
	}
	req := &pbNotification.JsonRequest{
		Details: details,
	}

	//Step 2 - send grpc message
	response, err := notificationApiStore[endPoint](context.Background(), req)

	//Step 3 - handle response
	if err != nil {
		zap.S().Errorf("error sending out nofitification. Endpoint: %s", endPoint)
	}
	return response, err
}
