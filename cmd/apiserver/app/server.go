// Copyright (c) 2021 Cisco Systems, Inc. and its affiliates
// All rights reserved
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

	DatasetsApiService := apiserver.NewDatasetsApiService()
	DatasetsApiController := openapi.NewDatasetsApiController(DatasetsApiService)

	DesignsApiService := apiserver.NewDesignsApiService()
	DesignsApiController := openapi.NewDesignsApiController(DesignsApiService)

	DesignCodesApiService := apiserver.NewDesignCodesApiService()
	DesignCodesApiController := openapi.NewDesignCodesApiController(DesignCodesApiService)

	DesignSchemasApiService := apiserver.NewDesignSchemasApiService()
	DesignSchemasApiController := openapi.NewDesignSchemasApiController(DesignSchemasApiService)

	// TODO remove me after prototyping phase is done.
	// only for prototyping
	DevServiceApi := apiserver.NewDevApiService()
	DevServiceApiController := openapi.NewDevApiController(DevServiceApi)

	JobsServiceApi := apiserver.NewJobsApiService()
	JobsServiceApiController := openapi.NewJobsApiController(JobsServiceApi)

	router := openapi.NewRouter(
		AgentServiceApiController,
		DatasetsApiController,
		DesignsApiController,
		DesignCodesApiController,
		DesignSchemasApiController,
		DevServiceApiController,
		JobsServiceApiController,
	)

	addr := fmt.Sprintf(":%d", portNo)
	zap.S().Fatal(http.ListenAndServe(addr, router))

	return nil
}
