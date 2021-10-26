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
	"encoding/json"
	"fmt"
	"io/ioutil"

	"github.com/cisco/fledge/cmd/fledgectl/resources"
	"github.com/cisco/fledge/pkg/openapi"
	"github.com/cisco/fledge/pkg/restapi"
)

type Params struct {
	resources.CommonParams

	JobFile string
	JobId   string
}

func Create(params Params) error {
	data, err := ioutil.ReadFile(params.JobFile)
	if err != nil {
		fmt.Printf("Failed to read file %s: %v\n", params.JobFile, err)
		return nil
	}

	// encode the data
	jobSpec := openapi.JobSpec{}
	err = json.Unmarshal(data, &jobSpec)
	if err != nil {
		fmt.Printf("Failed to parse %s\n", params.JobFile)
		return nil
	}

	// construct URL
	uriMap := map[string]string{
		"user": params.User,
	}
	url := restapi.CreateURL(params.Endpoint, restapi.CreateJobEndpoint, uriMap)

	// send post request
	code, resp, err := restapi.HTTPPost(url, jobSpec, "application/json")
	if err != nil || restapi.CheckStatusCode(code) != nil {
		fmt.Printf("Failed to create a job - code: %d, error: %v\n", code, err)
		return nil
	}

	jobStatus := openapi.JobStatus{}
	err = json.Unmarshal(resp, &jobStatus)
	if err != nil {
		fmt.Printf("WARNING: Failed to parse resp message: %v", err)
		return nil
	}

	fmt.Printf("New job created successfully\n")
	fmt.Printf("\tID: %s\n", jobStatus.Id)
	fmt.Printf("\tstate: %s\n", jobStatus.State)

	return nil
}

func Get(params Params) error {
	// TODO: implement me!
	fmt.Println("Not yet implemented")

	return nil
}

func GetStatus(params Params) error {
	// TODO: implement me!
	fmt.Println("Not yet implemented")

	return nil
}

func GetStatusMany(params Params) error {
	// TODO: implement me!
	fmt.Println("Not yet implemented")

	return nil
}

func Update(params Params) error {
	data, err := ioutil.ReadFile(params.JobFile)
	if err != nil {
		fmt.Printf("Failed to read file %s: %v\n", params.JobFile, err)
		return nil
	}

	// encode the data
	jobSpec := openapi.JobSpec{}
	err = json.Unmarshal(data, &jobSpec)
	if err != nil {
		fmt.Printf("Failed to parse %s\n", params.JobFile)
		return nil
	}

	// construct URL
	uriMap := map[string]string{
		"user":  params.User,
		"jobId": params.JobId,
	}
	url := restapi.CreateURL(params.Endpoint, restapi.UpdateJobEndPoint, uriMap)

	// send put request
	code, _, err := restapi.HTTPPut(url, jobSpec, "application/json")
	if err != nil || restapi.CheckStatusCode(code) != nil {
		fmt.Printf("Failed to update a job - code: %d, error: %v\n", code, err)
		return nil
	}

	fmt.Printf("Job updated successfully\n")

	return nil
}

func Remove(params Params) error {
	// TODO: implement me!
	fmt.Println("Not yet implemented")

	return nil
}

func Start(params Params) error {
	// construct URL
	uriMap := map[string]string{
		"user":  params.User,
		"jobId": params.JobId,
	}
	url := restapi.CreateURL(params.Endpoint, restapi.UpdateJobStatusEndPoint, uriMap)

	jobStatus := openapi.JobStatus{
		Id:    params.JobId,
		State: openapi.STARTING,
	}

	code, _, err := restapi.HTTPPut(url, jobStatus, "application/json")
	if err != nil || restapi.CheckStatusCode(code) != nil {
		fmt.Printf("Failed to start a job - code: %d, error: %v\n", code, err)
		return nil
	}

	fmt.Printf("Initiated to start a job successfully\n")

	return nil
}

func Stop(params Params) error {
	// TODO: implement me!
	fmt.Println("Not yet implemented")

	return nil
}
