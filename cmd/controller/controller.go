package main

import (
	"go.uber.org/zap"

	"wwwin-github.cisco.com/eti/fledge/cmd/controller/database"
	grpcctlr "wwwin-github.cisco.com/eti/fledge/cmd/controller/grpc"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

func main() {
	loggerMgr := util.InitZapLog()
	zap.ReplaceGlobals(loggerMgr)
	defer loggerMgr.Sync()

	//Init Database connection
	//todo remove hardcoded obj creation & remove db init from server.go file.
	database.NewDBService(database.DBInfo{Name: util.MONGODB, URI: "mongodb://localhost:27017"})

	//start GRPC service
	grpcctlr.StartGRPCService()
}
