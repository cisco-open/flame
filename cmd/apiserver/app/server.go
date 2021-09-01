package app

import (
	"fmt"
	"net/http"

	"go.uber.org/zap"

	"wwwin-github.cisco.com/eti/fledge/pkg/openapi"
	"wwwin-github.cisco.com/eti/fledge/pkg/openapi/apiserver"
)

//crate channel to listen for shutdown and other commands?
//https://medium.com/@pinkudebnath/graceful-shutdown-of-golang-servers-using-context-and-os-signals-cc1fa2c55e97

func RunServer(portNo uint16, ctlrInfo openapi.ServerInfo) error {
	zap.S().Infof("Staring API server. Controller %v", ctlrInfo)

	// Setting up REST API Server
	apiserver.Host = ctlrInfo.Ip
	apiserver.Port = uint16(ctlrInfo.Port)

	DesignApiService := apiserver.NewDesignApiService()
	DesignApiController := openapi.NewDesignApiController(DesignApiService)

	DesignsApiService := apiserver.NewDesignsApiService()
	DesignsApiController := openapi.NewDesignsApiController(DesignsApiService)

	DesignSchemaApiService := apiserver.NewDesignSchemaApiService()
	DesignSchemaApiController := openapi.NewDesignSchemaApiController(DesignSchemaApiService)

	JobServiceApi := apiserver.NewJobApiService()
	JobServiceApiController := openapi.NewJobApiController(JobServiceApi)

	JobsServiceApi := apiserver.NewJobsApiService()
	JobsServiceApiController := openapi.NewJobsApiController(JobsServiceApi)

	AgentServiceApi := apiserver.NewAgentApiService()
	AgentServiceApiController := openapi.NewAgentApiController(AgentServiceApi)

	// TODO remove me after prototyping phase is done.
	// only for prototyping
	DevServiceApi := apiserver.NewDevApiService()
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

	addr := fmt.Sprintf(":%d", portNo)
	zap.S().Fatal(http.ListenAndServe(addr, router))

	return nil
}
