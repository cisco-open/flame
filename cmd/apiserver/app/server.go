package app

import (
	"fmt"
	"net/http"

	klog "k8s.io/klog/v2"

	"wwwin-github.cisco.com/fledge/fledge/cmd/apiserver/openapi"
)

func RunServer(portNo uint16) error {
	klog.Infof("Staring a server on port %d", portNo)

	DesignSchemaApiService := openapi.NewDesignSchemaApiService()
	DesignSchemaApiController := openapi.NewDesignSchemaApiController(DesignSchemaApiService)

	DesignsApiService := openapi.NewDesignsApiService()
	DesignsApiController := openapi.NewDesignsApiController(DesignsApiService)

	router := openapi.NewRouter(DesignSchemaApiController, DesignsApiController)
	addr := fmt.Sprintf(":%d", portNo)
	klog.Fatal(http.ListenAndServe(addr, router))

	return nil
}
