package app

import (
	"fmt"
	"net/http"

	"go.uber.org/zap"
	"wwwin-github.cisco.com/fledge/fledge/cmd/apiserver/openapi"
)

//crate channel to listen for shutdown and other commands?
//https://medium.com/@pinkudebnath/graceful-shutdown-of-golang-servers-using-context-and-os-signals-cc1fa2c55e97
func RunServer(portNo uint16) error {
	zap.S().Infof("Staring API server ... | Port : %d", portNo)

	DesignsApiService := openapi.NewDesignApiService()
	DesignsApiController := openapi.NewDesignApiController(DesignsApiService)

	DesignSchemaApiService := openapi.NewDesignSchemaApiService()
	DesignSchemaApiController := openapi.NewDesignSchemaApiController(DesignSchemaApiService)

	router := openapi.NewRouter(DesignsApiController, DesignSchemaApiController)
	addr := fmt.Sprintf(":%d", portNo)
	zap.S().Fatal(http.ListenAndServe(addr, router))

	return nil
}
