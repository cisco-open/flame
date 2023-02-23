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

	"github.com/cisco-open/flame/cmd/flamectl/models"
	"github.com/cisco-open/flame/pkg/openapi"
	"github.com/cisco-open/flame/pkg/restapi"
)

func (c *container) Create(params *models.JobParams) error {
	data, err := os.ReadFile(params.File)
	if err != nil {
		fmt.Printf("Failed to read file %s: %v\n", params.File, err)
		return nil
	}

	// encode the data
	jobSpec := openapi.JobSpec{}
	err = json.Unmarshal(data, &jobSpec)
	if err != nil {
		fmt.Printf("Failed to parse %s\n", params.File)
		return nil
	}

	// construct URL
	uriMap := map[string]string{
		"user": params.User,
	}
	url := restapi.CreateURL(params.Endpoint, restapi.CreateJobEndpoint, uriMap)

	// send post request
	code, body, err := restapi.HTTPPost(url, jobSpec, "application/json")
	if err != nil || restapi.CheckStatusCode(code) != nil {
		var msg string
		_ = json.Unmarshal(body, &msg)
		fmt.Printf("Failed to create a job - code: %d; %s\n", code, msg)
		return nil
	}

	jobStatus := openapi.JobStatus{}
	err = json.Unmarshal(body, &jobStatus)
	if err != nil {
		fmt.Printf("WARNING: Failed to parse resp message: %v", err)
		return nil
	}

	fmt.Printf("New job created successfully\n")
	fmt.Printf("\tID: %s\n", jobStatus.Id)
	fmt.Printf("\tstate: %s\n", jobStatus.State)

	return nil
}
