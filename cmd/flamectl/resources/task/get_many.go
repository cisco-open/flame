// Copyright 2022 Cisco Systems, Inc. and its affiliates
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

package task

import (
	"encoding/json"
	"os"

	"github.com/olekukonko/tablewriter"
	"go.uber.org/zap"

	"github.com/cisco-open/flame/cmd/flamectl/models"
	"github.com/cisco-open/flame/pkg/openapi"
	"github.com/cisco-open/flame/pkg/restapi"
)

func (c *container) GetMany(params *models.TaskParams) error {
	logger := c.logger.Sugar().WithOptions(zap.Fields(
		zap.String("user", params.User),
		zap.String("jobID", params.JobId),
		zap.String("limit", params.Limit),
	))

	// construct URL
	uriMap := map[string]string{
		"user":  params.User,
		"jobId": params.JobId,
		"limit": params.Limit,
	}
	url := restapi.CreateURL(params.Endpoint, restapi.GetTasksInfoEndpoint, uriMap)

	code, responseBody, err := restapi.HTTPGet(url)
	if err != nil || restapi.CheckStatusCode(code) != nil {
		var msg string
		_ = json.Unmarshal(responseBody, &msg)
		logger.Errorf("Failed to retrieve tasks info - code: %d; %s\n", code, msg)
		return nil
	}

	// convert the response into list of struct
	infoList := []openapi.TaskInfo{}
	err = json.Unmarshal(responseBody, &infoList)
	if err != nil {
		logger.Errorf("Failed to unmarshal tasks info: %v\n", err)
		return nil
	}

	if len(infoList) == 0 {
		logger.Errorf("No info found for tasks of job %s. Job may not be ready or running.\n", params.JobId)
		return nil
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"Job ID", "Task ID", "Type", "State", "Timestamp"})

	for _, v := range infoList {
		table.Append([]string{v.JobId, v.TaskId, string(v.Type), string(v.State), v.Timestamp.String()})
	}

	table.Render() // Send output

	return nil
}
