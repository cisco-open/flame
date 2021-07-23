package app

import (
	"fmt"
	"net/http"

	"go.uber.org/zap"
	"wwwin-github.cisco.com/eti/fledge/cmd/apiserver/openapi"
)

//crate channel to listen for shutdown and other commands?
//https://medium.com/@pinkudebnath/graceful-shutdown-of-golang-servers-using-context-and-os-signals-cc1fa2c55e97
func RunServer(portNo uint16) error {
	zap.S().Infof("Staring API server ... | Port : %d", portNo)

	//Setting up REST API Server
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

	// TODO remove me after prototyping phase is done.
	// only for prototyping
	DevServiceApi := openapi.NewDevApiService()
	DevServiceApiController := openapi.NewDevApiController(DevServiceApi)

	router := openapi.NewRouter(DesignApiController, DesignsApiController, DesignSchemaApiController, JobServiceApiController, JobsServiceApiController, DevServiceApiController)
	addr := fmt.Sprintf(":%d", portNo)
	zap.S().Fatal(http.ListenAndServe(addr, router))

	return nil
}
