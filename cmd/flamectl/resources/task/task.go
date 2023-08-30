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
	"fmt"
	"os"

	"github.com/olekukonko/tablewriter"

	"github.com/cisco-open/flame/cmd/flamectl/resources"
	"github.com/cisco-open/flame/pkg/openapi"
	"github.com/cisco-open/flame/pkg/openapi/constants"
	"github.com/cisco-open/flame/pkg/restapi"
	"github.com/cisco-open/flame/pkg/util"
)

type Params struct {
	resources.CommonParams

	JobId  string
	TaskId string
	Limit  string
}

func Get(params Params) error {
	// construct URL
	uriMap := map[string]string{
		constants.ParamUser:   params.User,
		constants.ParamJobID:  params.JobId,
		constants.ParamTaskID: params.TaskId,
	}
	url := restapi.CreateURL(params.Endpoint, restapi.GetTaskInfoEndpoint, uriMap)

	code, responseBody, err := restapi.HTTPGet(url)
	if err != nil || restapi.CheckStatusCode(code) != nil {
		var msg string
		_ = json.Unmarshal(responseBody, &msg)
		fmt.Printf("Failed to retrieve task info - code: %d; %s\n", code, msg)
		return nil
	}

	taskInfoString, err := util.PrettyJsonString(responseBody)
	if err != nil {
		fmt.Printf("Failed to process task info: %v\n", err)
		return nil
	}

	fmt.Printf("%s\n", taskInfoString)

	return nil
}

func GetMany(params Params) error {
	// construct URL
	uriMap := map[string]string{
		constants.ParamUser:    params.User,
		constants.ParamJobID:   params.JobId,
		constants.ParamLimit:   params.Limit,
		constants.ParamGeneric: "false",
	}
	url := restapi.CreateURL(params.Endpoint, restapi.GetTasksInfoEndpoint, uriMap)

	code, responseBody, err := restapi.HTTPGet(url)
	if err != nil || restapi.CheckStatusCode(code) != nil {
		var msg string
		_ = json.Unmarshal(responseBody, &msg)
		fmt.Printf("Failed to retrieve tasks info - code: %d; %s\n", code, msg)
		return nil
	}

	// convert the response into list of struct
	infoList := []openapi.TaskInfo{}
	err = json.Unmarshal(responseBody, &infoList)
	if err != nil {
		fmt.Printf("Failed to unmarshal tasks info: %v\n", err)
		return nil
	}

	if len(infoList) == 0 {
		fmt.Printf("No info found for tasks of job %s. Job may not be ready or running.\n", params.JobId)
		return nil
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"Job ID", "Task ID", "State", "Timestamp"})

	for _, v := range infoList {
		table.Append([]string{v.JobId, v.TaskId, string(v.State), v.Timestamp.String()})
	}

	table.Render() // Send output

	return nil
}
