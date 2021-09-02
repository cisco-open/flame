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

	// setting controller IP and port number for rest api call to the controller from the api server
	apiserver.Host = ctlrInfo.Ip
	apiserver.Port = uint16(ctlrInfo.Port)

	// Setting up REST API Server
	AgentServiceApi := apiserver.NewAgentApiService()
	AgentServiceApiController := openapi.NewAgentApiController(AgentServiceApi)

	DesignCodesApiService := apiserver.NewDesignCodesApiService()
	DesignCodesApiController := openapi.NewDesignCodesApiController(DesignCodesApiService)

	DesignSchemasApiService := apiserver.NewDesignSchemasApiService()
	DesignSchemasApiController := openapi.NewDesignSchemasApiController(DesignSchemasApiService)

	DesignsApiService := apiserver.NewDesignsApiService()
	DesignsApiController := openapi.NewDesignsApiController(DesignsApiService)

	// TODO remove me after prototyping phase is done.
	// only for prototyping
	DevServiceApi := apiserver.NewDevApiService()
	DevServiceApiController := openapi.NewDevApiController(DevServiceApi)

	JobServiceApi := apiserver.NewJobApiService()
	JobServiceApiController := openapi.NewJobApiController(JobServiceApi)

	JobsServiceApi := apiserver.NewJobsApiService()
	JobsServiceApiController := openapi.NewJobsApiController(JobsServiceApi)

	router := openapi.NewRouter(
		AgentServiceApiController,
		DesignsApiController,
		DesignCodesApiController,
		DesignSchemasApiController,
		DevServiceApiController,
		JobServiceApiController,
		JobsServiceApiController,
	)

	addr := fmt.Sprintf(":%d", portNo)
	zap.S().Fatal(http.ListenAndServe(addr, router))

	return nil
}
