package main

import (
	"go.uber.org/zap"

	grpcnotify "wwwin-github.cisco.com/eti/fledge/cmd/notificationservice/grpc"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

func main() {
	loggerMgr := util.InitZapLog()
	zap.ReplaceGlobals(loggerMgr)
	defer loggerMgr.Sync()

	//start GRPC service
	grpcnotify.StartGRPCService(util.NotificationServiceGrpcPort)
}
