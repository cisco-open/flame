// Copyright (c) 2022 Cisco Systems, Inc. and its affiliates
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

package task

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/olekukonko/tablewriter"

	"github.com/cisco-open/flame/cmd/flamectl/resources"
	"github.com/cisco-open/flame/pkg/openapi"
	"github.com/cisco-open/flame/pkg/restapi"
	"github.com/cisco-open/flame/pkg/util"
)

type Params struct {
	resources.CommonParams

	JobId string
	Limit string
}

func GetMany(params Params) error {
	// construct URL
	uriMap := map[string]string{
		"user":  params.User,
		"jobId": params.JobId,
		"limit": params.Limit,
	}
	url := restapi.CreateURL(params.Endpoint, restapi.GetTasksInfoEndpoint, uriMap)

	code, responseBody, err := restapi.HTTPGet(url)
	if err != nil || restapi.CheckStatusCode(code) != nil {
		var errMsg error
		_ = util.ByteToStruct(responseBody, &errMsg)
		fmt.Printf("Failed to retrieve tasks info - code: %d, error: %v, msg: %v\n", code, err, errMsg)
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
	table.SetHeader([]string{"Job ID", "Agent ID", "Type", "State", "Timestamp"})

	for _, v := range infoList {
		table.Append([]string{v.JobId, v.AgentId, string(v.Type), string(v.State), v.Timestamp.String()})
	}

	table.Render() // Send output

	return nil
}
