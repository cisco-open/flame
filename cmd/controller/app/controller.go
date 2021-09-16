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
	"wwwin-github.cisco.com/eti/fledge/cmd/controller/app/jobeventq"
	"wwwin-github.cisco.com/eti/fledge/pkg/openapi"
	"wwwin-github.cisco.com/eti/fledge/pkg/openapi/controller"
)

type Controller struct {
	dbUri      string
	notifierEp string
	restPort   string

	jobEventQ *jobeventq.EventQ
}

func NewController(dbUri string, notifierEp string, restPort string) (*Controller, error) {
	jobEventQ := jobeventq.NewEventQ(0)
	if jobEventQ == nil {
		return nil, fmt.Errorf("failed to create a job event queue")
	}

	instance := &Controller{
		dbUri:      dbUri,
		notifierEp: notifierEp,
		restPort:   restPort,

		jobEventQ: jobEventQ,
	}

	return instance, nil
}

func (c *Controller) Start() {
	zap.S().Infof("Connecting to database at %s", c.dbUri)
	err := database.NewDBService(c.dbUri)
	if err != nil {
		zap.S().Fatalf("Failed to connect to DB: %v", err)
	}

	jobMgr, err := NewJobManager(c.jobEventQ)
	if err != nil {
		zap.S().Fatalf("Failed to create a job manager: %v", err)
	}

	// GRPC server & connections
	grpcctlr.InitGRPCService(c.notifierEp)

	go jobMgr.Do()

	go c.serveRestApi()

	select {}
}

func (c *Controller) serveRestApi() {
	// Setting up REST API Server
	zap.S().Infof("Staring controller... | Port: %s", c.restPort)

	apiRouters := []openapi.Router{
		openapi.NewDatasetsApiController(controller.NewDatasetsApiService()),
		openapi.NewDesignsApiController(controller.NewDesignsApiService()),
		openapi.NewDesignCodesApiController(controller.NewDesignCodesApiService()),
		openapi.NewDesignSchemasApiController(controller.NewDesignSchemasApiService()),
		openapi.NewJobsApiController(controller.NewJobsApiService(c.jobEventQ)),
		openapi.NewAgentApiController(controller.NewAgentApiService()),
		// TODO: remove me after prototyping phase is done; it's only for prototyping
		openapi.NewDevApiController(controller.NewDevApiService()),
	}
	controller.CacheInit()

	router := openapi.NewRouter(apiRouters...)

	zap.S().Fatal(http.ListenAndServe(":"+c.restPort, router))
}
