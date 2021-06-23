package app

import (
	"fmt"
	"net/http"

	"go.uber.org/zap"
	"wwwin-github.cisco.com/fledge/fledge/cmd/apiserver/openapi"
	"wwwin-github.cisco.com/fledge/fledge/cmd/controller/database"
)

//crate channel to listen for shutdown and other commands?
//https://medium.com/@pinkudebnath/graceful-shutdown-of-golang-servers-using-context-and-os-signals-cc1fa2c55e97
func RunServer(portNo uint16, dbObj database.DBInfo) error {
	zap.S().Infof("Staring API server ... | Port : %d", portNo)

	//Init Database connection
	//todo move this under controller initialization.
	database.NewDBService(dbObj)

	//Setting up REST API Server
	DesignApiService := openapi.NewDesignApiService()
	DesignApiController := openapi.NewDesignApiController(DesignApiService)

	DesignsApiService := openapi.NewDesignsApiService()
	DesignsApiController := openapi.NewDesignsApiController(DesignsApiService)

	DesignSchemaApiService := openapi.NewDesignSchemaApiService()
	DesignSchemaApiController := openapi.NewDesignSchemaApiController(DesignSchemaApiService)

	router := openapi.NewRouter(DesignApiController, DesignsApiController, DesignSchemaApiController)
	addr := fmt.Sprintf(":%d", portNo)
	zap.S().Fatal(http.ListenAndServe(addr, router))

	return nil
}
