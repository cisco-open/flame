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

// InitGRPCService starts the controller grpc server and establishes connection with the notification service
func InitGRPCService(notificationServer objects.ServerInfo) {
	ControllerGRPC.connectToNotificationService(notificationServer)
}
