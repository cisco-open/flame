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

	"wwwin-github.cisco.com/eti/fledge/cmd/controller/app/database"
	grpcctlr "wwwin-github.cisco.com/eti/fledge/cmd/controller/app/grpc"
	"wwwin-github.cisco.com/eti/fledge/pkg/openapi"
	"wwwin-github.cisco.com/eti/fledge/pkg/openapi/controller"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

func StartController(uri string, nsInfo openapi.ServerInfo) error {
	err := connectDB(uri)
	if err != nil {
		zap.S().Fatalf("Failed to connect to DB: %v", err)
	}

	connectGRPC(nsInfo)

	startServer()

	return nil
}

//connectDB initialize and connect to the database connection
func connectDB(uri string) error {
	zap.S().Infof("Connecting to database at %s", uri)
	return database.NewDBService(uri)
}

func connectGRPC(nsInfo openapi.ServerInfo) {
	//GRPC server & connections
	grpcctlr.InitGRPCService(nsInfo)
}

func startServer() {
	// Setting up REST API Server
	zap.S().Infof("Staring controller ... | Port : %d", util.ControllerRestApiPort)

	DatasetsApiService := controller.NewDatasetsApiService()
	DatasetsApiController := openapi.NewDatasetsApiController(DatasetsApiService)

	DesignsApiService := controller.NewDesignsApiService()
	DesignsApiController := openapi.NewDesignsApiController(DesignsApiService)

	DesignCodesApiService := controller.NewDesignCodesApiService()
	DesignCodesApiController := openapi.NewDesignCodesApiController(DesignCodesApiService)

	DesignSchemasApiService := controller.NewDesignSchemasApiService()
	DesignSchemasApiController := openapi.NewDesignSchemasApiController(DesignSchemasApiService)

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
		DatasetsApiController,
		DesignsApiController,
		DesignCodesApiController,
		DesignSchemasApiController,
		JobsServiceApiController,
		DevServiceApiController,
		AgentServiceApiController,
	)

	addr := fmt.Sprintf(":%d", util.ControllerRestApiPort)
	zap.S().Fatal(http.ListenAndServe(addr, router))
}
