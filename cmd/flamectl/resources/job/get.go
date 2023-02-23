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

	"github.com/cisco-open/flame/cmd/flamectl/models"
	"github.com/cisco-open/flame/pkg/openapi"
	"github.com/cisco-open/flame/pkg/restapi"
	"github.com/cisco-open/flame/pkg/util"
)

func (c *container) Get(params *models.JobParams) error {
	// construct URL
	uriMap := map[string]string{
		"user":  params.User,
		"jobId": params.Id,
	}
	url := restapi.CreateURL(params.Endpoint, restapi.GetJobEndPoint, uriMap)

	code, body, err := restapi.HTTPGet(url)
	if err != nil || restapi.CheckStatusCode(code) != nil {
		var msg string
		_ = json.Unmarshal(body, &msg)
		fmt.Printf("Failed to Get job - code: %d; %s\n", code, msg)
		return nil
	}

	fmt.Printf("Request to get job successful\n")

	// convert the response into list of struct
	jobSpec := openapi.JobSpec{}
	err = json.Unmarshal(body, &jobSpec)
	if err != nil {
		fmt.Printf("Failed to unmarshal get job: %v\n", err)
		return nil
	}

	bodyJson, err := util.JSONMarshal(jobSpec)
	if err != nil {
		fmt.Printf("WARNING: error while marshaling json: %v\n\n", err)
		fmt.Println(string(body))
	}

	prettyJSON, err := util.FormatJSON(bodyJson)
	if err != nil {
		fmt.Printf("WARNING: error while formating json: %v\n\n", err)
		fmt.Println(string(body))
	} else {
		fmt.Println(string(prettyJSON))
	}

	return nil
}
