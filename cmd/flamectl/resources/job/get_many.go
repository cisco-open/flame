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

package job

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/olekukonko/tablewriter"

	"github.com/cisco-open/flame/cmd/flamectl/models"
	"github.com/cisco-open/flame/pkg/openapi"
	"github.com/cisco-open/flame/pkg/restapi"
)

func (c *container) GetMany(params *models.JobParams) error {
	// construct URL
	uriMap := map[string]string{
		"user":  params.User,
		"limit": params.Limit,
	}
	url := restapi.CreateURL(params.Endpoint, restapi.GetJobsEndPoint, uriMap)

	code, body, err := restapi.HTTPGet(url)
	if err != nil || restapi.CheckStatusCode(code) != nil {
		var msg string
		_ = json.Unmarshal(body, &msg)
		fmt.Printf("Failed to retrieve jobs' status - code: %d; %s\n", code, msg)
		return nil
	}

	// convert the response into list of struct
	infoList := []openapi.JobStatus{}
	err = json.Unmarshal(body, &infoList)
	if err != nil {
		fmt.Printf("Failed to unmarshal job status: %v\n", err)
		return nil
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"Job ID", "State", "created At", "started At", "ended At"})

	for _, v := range infoList {
		table.Append([]string{v.Id, string(v.State), v.CreatedAt.String(), v.StartedAt.String(), v.EndedAt.String()})
	}

	table.Render() // Send output

	return nil
}
