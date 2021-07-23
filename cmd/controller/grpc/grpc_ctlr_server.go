package grpcctlr

import (
	"wwwin-github.cisco.com/eti/fledge/pkg/objects"
	pbNotification "wwwin-github.cisco.com/eti/fledge/pkg/proto/go/notification"
)

var ControllerGRPC = &controllerGRPC{}

//controllerGRPC implement the controller grpc server which is used by the REST API to send user requests
//also maintains connection with other services example: notification service.
type controllerGRPC struct {
	notificationServiceClient pbNotification.NotificationControllerStoreClient
}

//startGRPCServer grpc server used by the REST API
/* TODO - keep it for now. But to be removed if we are sure we won't be using GRPC for any other purpose.
func startGRPCServer() {
	lis, err := net.Listen("tcp", ":"+strconv.Itoa(util.ControllerGrpcPort))
	if err != nil {
		zap.S().Errorf("failed to listen grpc server: %v", err)
	}

	// create grpc server
	s := grpc.NewServer()

	zap.S().Infof("Controller GRPC server listening at %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		zap.S().Errorf("failed to serve: %s", err)
	}
}*/

// InitGRPCService starts the controller grpc server and establishes connection with the notification service
func InitGRPCService(notificationServer objects.ServerInfo) {
	ControllerGRPC.connectToNotificationService(notificationServer)
}
