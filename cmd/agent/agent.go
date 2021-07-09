package main

import (
	"go.uber.org/zap"

	grpcagent "wwwin-github.cisco.com/eti/fledge/cmd/agent/grpc"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

func main() {
	loggerMgr := util.InitZapLog(util.Agent)
	zap.ReplaceGlobals(loggerMgr)
	defer loggerMgr.Sync()

	grpcagent.ConnectToNotificationService("localhost", util.NotificationServiceGrpcPort)
}
