package openapi

import (
	"fmt"
	"net/http"

	"go.uber.org/zap"
	"wwwin-github.cisco.com/eti/fledge/pkg/objects"
)

//crate channel to listen for shutdown and other commands?
//https://medium.com/@pinkudebnath/graceful-shutdown-of-golang-servers-using-context-and-os-signals-cc1fa2c55e97

var App AppInfo

func RunServer(portNo uint16, ctlrInfo objects.ServerInfo) error {
	App = AppInfo{controllerInfo: ctlrInfo}
	zap.S().Infof("Staring API server. Controller %v", ctlrInfo)

	//Setting up REST API Server
	DesignApiService := NewDesignApiService()
	DesignApiController := NewDesignApiController(DesignApiService)

	DesignsApiService := NewDesignsApiService()
	DesignsApiController := NewDesignsApiController(DesignsApiService)

	DesignSchemaApiService := NewDesignSchemaApiService()
	DesignSchemaApiController := NewDesignSchemaApiController(DesignSchemaApiService)

	JobServiceApi := NewJobApiService()
	JobServiceApiController := NewJobApiController(JobServiceApi)

	JobsServiceApi := NewJobsApiService()
	JobsServiceApiController := NewJobsApiController(JobsServiceApi)

	// TODO remove me after prototyping phase is done.
	// only for prototyping
	DevServiceApi := NewDevApiService()
	DevServiceApiController := NewDevApiController(DevServiceApi)

	router := NewRouter(DesignApiController, DesignsApiController, DesignSchemaApiController, JobServiceApiController, JobsServiceApiController, DevServiceApiController)
	addr := fmt.Sprintf(":%d", portNo)
	zap.S().Fatal(http.ListenAndServe(addr, router))

	return nil
}
