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
	apiserver.HostEndpoint = fmt.Sprintf("%s:%d", ctlrInfo.Ip, ctlrInfo.Port)

	// Setting up REST API Server
	apiRouters := []openapi.Router{
		openapi.NewAgentApiController(apiserver.NewAgentApiService()),
		openapi.NewDatasetsApiController(apiserver.NewDatasetsApiService()),
		openapi.NewDesignsApiController(apiserver.NewDesignsApiService()),
		openapi.NewDesignCodesApiController(apiserver.NewDesignCodesApiService()),
		openapi.NewDesignSchemasApiController(apiserver.NewDesignSchemasApiService()),
		// TODO: remove me after prototyping phase is done. it's only for prototyping.
		openapi.NewDevApiController(apiserver.NewDevApiService()),
		openapi.NewJobsApiController(apiserver.NewJobsApiService()),
	}

	router := openapi.NewRouter(apiRouters...)

	addr := fmt.Sprintf(":%d", portNo)
	zap.S().Fatal(http.ListenAndServe(addr, router))

	return nil
}
