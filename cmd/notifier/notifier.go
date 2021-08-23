package main

import (
	"go.uber.org/zap"

	grpcnotify "wwwin-github.cisco.com/eti/fledge/cmd/notifier/grpc"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

func main() {
	loggerMgr := util.InitZapLog(util.Notifier)
	zap.ReplaceGlobals(loggerMgr)
	defer loggerMgr.Sync()

	//start GRPC service
	grpcnotify.StartGRPCService(util.NotificationServiceGrpcPort)
}
