package app

import (
	"fmt"
	"net/http"

	"go.uber.org/zap"

	"wwwin-github.cisco.com/eti/fledge/cmd/controller/app/database"
	grpcctlr "wwwin-github.cisco.com/eti/fledge/cmd/controller/app/grpc"
	"wwwin-github.cisco.com/eti/fledge/pkg/openapi"
	"wwwin-github.cisco.com/eti/fledge/pkg/openapi/controller"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

func StartController(uri string, nsInfo openapi.ServerInfo) error {
	connectDB(uri)
	connectGRPC(nsInfo)
	startServer()

	return nil
}

//connectDB initialize and connect to the database connection
func connectDB(uri string) {
	zap.S().Infof("Connecting to database at %s", uri)
	database.NewDBService(uri)
}

func connectGRPC(nsInfo openapi.ServerInfo) {
	//GRPC server & connections
	grpcctlr.InitGRPCService(nsInfo)
}

func startServer() {
	//Setting up REST API Server
	zap.S().Infof("Staring controller ... | Port : %d", util.ControllerRestApiPort)
	DesignApiService := controller.NewDesignApiService()
	DesignApiController := openapi.NewDesignApiController(DesignApiService)

	DesignsApiService := controller.NewDesignsApiService()
	DesignsApiController := openapi.NewDesignsApiController(DesignsApiService)

	DesignSchemaApiService := controller.NewDesignSchemaApiService()
	DesignSchemaApiController := openapi.NewDesignSchemaApiController(DesignSchemaApiService)

	JobServiceApi := controller.NewJobApiService()
	JobServiceApiController := openapi.NewJobApiController(JobServiceApi)

	JobsServiceApi := controller.NewJobsApiService()
	JobsServiceApiController := openapi.NewJobsApiController(JobsServiceApi)

	AgentServiceApi := controller.NewAgentApiService()
	AgentServiceApiController := openapi.NewAgentApiController(AgentServiceApi)

	controller.CacheInit()

	// TODO remove me after prototyping phase is done.
	// only for prototyping
	DevServiceApi := controller.NewDevApiService()
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
