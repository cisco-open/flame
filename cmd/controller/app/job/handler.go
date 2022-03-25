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

package job

import (
	"fmt"
	"sync"

	"github.com/cisco-open/flame/cmd/controller/app/database"
	"github.com/cisco-open/flame/cmd/controller/config"
	"github.com/cisco-open/flame/pkg/util"
)

const (
	deploymentDirPath     = "/" + util.ProjectName + "/deployment"
	deploymentTemplateDir = "templates"

	jobDeploymentFilePrefix = "job-agent"
	jobTemplatePath         = deploymentDirPath + "/" + jobDeploymentFilePrefix + ".yaml.mustache"

	defaultHandler = "default"
)

var (
	helmChartFiles = []string{"Chart.yaml", "values.yaml"}
)

type handler interface {
	Do()
}

func NewHandler(handlerType string, dbService database.DBService, jobId string,
	eventQ *EventQ, jobQueues map[string]*EventQ, mu *sync.Mutex,
	notifier string, jobParams config.JobParams, platform string) (handler, error) {
	var hdlr handler
	var err error

	switch handlerType {
	case defaultHandler:
		hdlr, err = NewDefaultHandler(dbService, jobId, eventQ, jobQueues, mu, notifier, jobParams, platform)

	default:
		err = fmt.Errorf("unknown handler type")
	}

	if err != nil {
		return nil, err
	}

	return hdlr, nil
}
