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

	"github.com/cisco-open/flame/cmd/controller/app/database"
	"github.com/cisco-open/flame/cmd/controller/app/job"
	"github.com/cisco-open/flame/cmd/controller/config"
	"github.com/cisco-open/flame/pkg/openapi"
	"github.com/cisco-open/flame/pkg/openapi/controller"
)

type Controller struct {
	dbUri     string
	notifier  string
	restPort  string
	platform  string
	dbService database.DBService
	jobParams config.JobParams

	jobEventQ  *job.EventQ
	jobBuilder *job.JobBuilder
}

func NewController(cfg *config.Config) (*Controller, error) {
	zap.S().Infof("Connecting to database at %s", cfg.Db)
	dbService, err := database.NewDBService(cfg.Db)
	if err != nil {
		return nil, err
	}

	jobEventQ := job.NewEventQ(0)
	if jobEventQ == nil {
		return nil, fmt.Errorf("failed to create a job event queue")
	}

	jobBuilder := job.NewJobBuilder(dbService, cfg.JobParams)
	if jobBuilder == nil {
		return nil, fmt.Errorf("failed to create a job builder")
	}

	instance := &Controller{
		dbUri:     cfg.Db,
		notifier:  cfg.Notifier,
		restPort:  cfg.Port,
		platform:  cfg.Platform,
		dbService: dbService,
		jobParams: cfg.JobParams,

		jobEventQ:  jobEventQ,
		jobBuilder: jobBuilder,
	}

	return instance, nil
}

func (c *Controller) Start() {
	jobMgr, err := job.NewManager(c.dbService, c.jobEventQ, c.notifier, c.jobParams, c.platform)
	if err != nil {
		zap.S().Fatalf("Failed to create a job manager: %v", err)
	}

	go jobMgr.Do()

	go c.serveRestApi()

	select {}
}

func (c *Controller) serveRestApi() {
	// Setting up REST API Server
	zap.S().Infof("Staring controller... | Port: %s", c.restPort)

	apiRouters := []openapi.Router{
		openapi.NewDatasetsApiController(controller.NewDatasetsApiService(c.dbService)),
		openapi.NewDesignsApiController(controller.NewDesignsApiService(c.dbService)),
		openapi.NewDesignCodesApiController(controller.NewDesignCodesApiService(c.dbService)),
		openapi.NewDesignSchemasApiController(controller.NewDesignSchemasApiService(c.dbService)),
		openapi.NewJobsApiController(controller.NewJobsApiService(c.dbService, c.jobEventQ, c.jobBuilder)),
	}

	router := openapi.NewRouter(apiRouters...)

	zap.S().Fatal(http.ListenAndServe(":"+c.restPort, router))
}
