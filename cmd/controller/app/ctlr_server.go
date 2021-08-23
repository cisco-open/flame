package app

import (
	"fmt"
	"net/http"

	"go.uber.org/zap"

	"wwwin-github.cisco.com/eti/fledge/cmd/controller/database"
	grpcctlr "wwwin-github.cisco.com/eti/fledge/cmd/controller/grpc"
	"wwwin-github.cisco.com/eti/fledge/cmd/controller/openapi"
	"wwwin-github.cisco.com/eti/fledge/pkg/objects"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

func StartController(dbInfo database.DBInfo, nsInfo objects.ServerInfo) error {
	connectDB(dbInfo)
	connectGRPC(nsInfo)
	startServer()

	return nil
}

//connectDB initialize and connect to the database connection
func connectDB(info database.DBInfo) {
	zap.S().Infof("Connecting to %s database at %v", info.Name, info.URI)
	database.NewDBService(info)
}

func connectGRPC(nsInfo objects.ServerInfo) {
	//GRPC server & connections
	grpcctlr.InitGRPCService(nsInfo)
}

func startServer() {
	//Setting up REST API Server
	zap.S().Infof("Staring controller ... | Port : %d", util.ControllerRestApiPort)
	DesignApiService := openapi.NewDesignApiService()
	DesignApiController := openapi.NewDesignApiController(DesignApiService)

	DesignsApiService := openapi.NewDesignsApiService()
	DesignsApiController := openapi.NewDesignsApiController(DesignsApiService)

	DesignSchemaApiService := openapi.NewDesignSchemaApiService()
	DesignSchemaApiController := openapi.NewDesignSchemaApiController(DesignSchemaApiService)

	JobServiceApi := openapi.NewJobApiService()
	JobServiceApiController := openapi.NewJobApiController(JobServiceApi)

	JobsServiceApi := openapi.NewJobsApiService()
	JobsServiceApiController := openapi.NewJobsApiController(JobsServiceApi)

	AgentServiceApi := openapi.NewAgentApiService()
	AgentServiceApiController := openapi.NewAgentApiController(AgentServiceApi)

	openapi.CacheInit()

	// TODO remove me after prototyping phase is done.
	// only for prototyping
	DevServiceApi := openapi.NewDevApiService()
	DevServiceApiController := openapi.NewDevApiController(DevServiceApi)

	router := openapi.NewRouter(
		DesignApiController,
		DesignsApiController,
		DesignSchemaApiController,
		JobServiceApiController,
		JobsServiceApiController,
		DevServiceApiController,
		AgentServiceApiController,
	)
	addr := fmt.Sprintf(":%d", util.ControllerRestApiPort)
	zap.S().Fatal(http.ListenAndServe(addr, router))
}
