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
	"path"

	"github.com/olekukonko/tablewriter"

	"github.com/cisco-open/flame/cmd/flamectl/resources"
	"github.com/cisco-open/flame/pkg/openapi"
	"github.com/cisco-open/flame/pkg/openapi/constants"
	"github.com/cisco-open/flame/pkg/restapi"
	"github.com/cisco-open/flame/pkg/util"
)

type Params struct {
	resources.CommonParams

	JobFile string
	JobId   string
	Limit   string
}

type CreateJobRequest struct {
	Id            string              `json:"id,omitempty"`
	UserId        string              `json:"userId,omitempty"`
	DesignId      string              `json:"designId"`
	SchemaVersion string              `json:"schemaVersion"`
	CodeVersion   string              `json:"codeVersion"`
	Priority      openapi.JobPriority `json:"priority,omitempty"`
	MaxRunTime    int32               `json:"maxRunTime,omitempty"`
	Backend       openapi.CommBackend `json:"backend,omitempty"`
	DataSpecPath  string              `json:"dataSpecPath,omitempty"`
	ModelSpecPath string              `json:"modelSpecPath,omitempty"`
}

func createJobSpec(data []byte, jobFile string) (bool, openapi.JobSpec) {
	// encode the data
	createJobRequest := CreateJobRequest{}
	err := json.Unmarshal(data, &createJobRequest)
	if err != nil {
		fmt.Println("Failed to unmarshal the job file")
		return false, openapi.JobSpec{}
	}

	//validate data spec
	dataSpec := openapi.DataSpec{}
	dataSpecPath := fmt.Sprintf("%s/%s", path.Dir(jobFile), createJobRequest.DataSpecPath)

	err = util.ReadFileToStruct(dataSpecPath, &dataSpec)
	if err != nil {
		fmt.Println(err)
		fmt.Printf("Failed to parse dataSpec file %s\n", dataSpecPath)
		return false, openapi.JobSpec{}
	}

	//validate model spec
	modelSpec := openapi.ModelSpec{}
	modelSpecPath := fmt.Sprintf("%s/%s", path.Dir(jobFile), createJobRequest.ModelSpecPath)

	err = util.ReadFileToStruct(modelSpecPath, &modelSpec)
	if err != nil {
		fmt.Printf("Failed to parse modelSpec file %s\n", modelSpecPath)
		return false, openapi.JobSpec{}
	}

	jobSpec := openapi.JobSpec{
		Id:            createJobRequest.Id,
		UserId:        createJobRequest.UserId,
		DesignId:      createJobRequest.DesignId,
		SchemaVersion: createJobRequest.SchemaVersion,
		CodeVersion:   createJobRequest.CodeVersion,
		Priority:      createJobRequest.Priority,
		MaxRunTime:    createJobRequest.MaxRunTime,
		Backend:       createJobRequest.Backend,
		DataSpec:      dataSpec,
		ModelSpec:     modelSpec,
	}
	return true, jobSpec
}

func Create(params Params) error {
	data, err := os.ReadFile(params.JobFile)
	if err != nil {
		fmt.Printf("Failed to read file %s: %v\n", params.JobFile, err)
		return nil
	}

	// encode the data
	//validate the input - to ensure dataSpecPath and ModelSpecPath are correctly provide
	isValid, jobSpec := createJobSpec(data, params.JobFile)
	if !isValid {
		fmt.Printf("Incorrect job specification provided %s\n", params.JobFile)
		return nil
	}

	// construct URL
	uriMap := map[string]string{
		constants.ParamUser: params.User,
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

func Get(params Params) error {
	// construct URL
	uriMap := map[string]string{
		constants.ParamUser:  params.User,
		constants.ParamJobID: params.JobId,
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

func GetMany(params Params) error {
	// construct URL
	uriMap := map[string]string{
		constants.ParamUser:  params.User,
		constants.ParamLimit: params.Limit,
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

func GetStatus(params Params) error {
	// construct URL
	uriMap := map[string]string{
		constants.ParamUser:  params.User,
		constants.ParamJobID: params.JobId,
	}
	url := restapi.CreateURL(params.Endpoint, restapi.GetJobStatusEndPoint, uriMap)

	code, body, err := restapi.HTTPGet(url)
	if err != nil || restapi.CheckStatusCode(code) != nil {
		var msg string
		_ = json.Unmarshal(body, &msg)
		fmt.Printf("Failed to Get job status - code: %d; %s\n", code, msg)
		return nil
	}

	fmt.Printf("Request to get job status successful\n")

	// convert the response into list of struct
	jobStatus := openapi.JobStatus{}
	err = json.Unmarshal(body, &jobStatus)
	if err != nil {
		fmt.Printf("Failed to unmarshal get job status: %v\n", err)
		return nil
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"Job ID", "State", "created At", "started At", "ended At"})
	table.Append([]string{jobStatus.Id, string(jobStatus.State), jobStatus.CreatedAt.String(),
		jobStatus.StartedAt.String(), jobStatus.EndedAt.String()})
	table.Render() // Send output

	return nil
}

func Update(params Params) error {
	data, err := os.ReadFile(params.JobFile)
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
		constants.ParamUser:  params.User,
		constants.ParamJobID: params.JobId,
	}
	url := restapi.CreateURL(params.Endpoint, restapi.UpdateJobEndPoint, uriMap)

	// send put request
	code, body, err := restapi.HTTPPut(url, jobSpec, "application/json")
	if err != nil || restapi.CheckStatusCode(code) != nil {
		var msg string
		_ = json.Unmarshal(body, &msg)
		fmt.Printf("Failed to update a job - code: %d; %s\n", code, msg)
		return nil
	}

	fmt.Printf("Job updated successfully\n")

	return nil
}

func Remove(params Params) error {
	// construct URL
	uriMap := map[string]string{
		constants.ParamUser:  params.User,
		constants.ParamJobID: params.JobId,
	}
	url := restapi.CreateURL(params.Endpoint, restapi.DeleteJobEndPoint, uriMap)

	jobStatus := openapi.JobStatus{
		Id: params.JobId,
	}

	code, body, err := restapi.HTTPDelete(url, jobStatus, "application/json")
	if err != nil || restapi.CheckStatusCode(code) != nil {
		var msg string
		_ = json.Unmarshal(body, &msg)
		fmt.Printf("Failed to delete a job - code: %d; %s\n", code, msg)
		return nil
	}

	fmt.Printf("Request to delete a job successful\n")

	return nil
}

func Start(params Params) error {
	// construct URL
	uriMap := map[string]string{
		constants.ParamUser:  params.User,
		constants.ParamJobID: params.JobId,
	}
	url := restapi.CreateURL(params.Endpoint, restapi.UpdateJobStatusEndPoint, uriMap)

	jobStatus := openapi.JobStatus{
		Id:    params.JobId,
		State: openapi.STARTING,
	}

	code, body, err := restapi.HTTPPut(url, jobStatus, "application/json")
	if err != nil || restapi.CheckStatusCode(code) != nil {
		var msg string
		_ = json.Unmarshal(body, &msg)
		fmt.Printf("Failed to start a job - code: %d; %s\n", code, msg)
		return nil
	}

	fmt.Printf("Initiated to start a job successfully\n")

	return nil
}

func Stop(params Params) error {
	// construct URL
	uriMap := map[string]string{
		constants.ParamUser:  params.User,
		constants.ParamJobID: params.JobId,
	}
	url := restapi.CreateURL(params.Endpoint, restapi.UpdateJobStatusEndPoint, uriMap)

	jobStatus := openapi.JobStatus{
		Id:    params.JobId,
		State: openapi.STOPPING,
	}

	code, body, err := restapi.HTTPPut(url, jobStatus, "application/json")
	if err != nil || restapi.CheckStatusCode(code) != nil {
		var msg string
		_ = json.Unmarshal(body, &msg)
		fmt.Printf("Failed to stop a job - code: %d; %s\n", code, msg)
		return nil
	}

	fmt.Printf("Request to stop a job successful\n")

	return nil
}
