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

package dataset

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"

	"github.com/cisco-open/flame/cmd/flamectl/resources"
	"github.com/cisco-open/flame/pkg/openapi"
	"github.com/cisco-open/flame/pkg/restapi"
	"github.com/olekukonko/tablewriter"
)

type Params struct {
	resources.CommonParams

	DatasetFile string
	DatasetId   string
	Limit       string
}

func Create(params Params) error {
	data, err := os.ReadFile(params.DatasetFile)
	if err != nil {
		fmt.Printf("Failed to read file %s: %v\n", params.DatasetFile, err)
		return nil
	}

	// Encode the data
	datasetInfo := openapi.DatasetInfo{}
	err = json.Unmarshal(data, &datasetInfo)
	if err != nil {
		fmt.Printf("Failed to unmarshal: %v\n", err)
		return nil
	}

	// construct URL
	uriMap := map[string]string{
		"user": params.User,
	}
	url := restapi.CreateURL(params.Endpoint, restapi.CreateDatasetEndPoint, uriMap)

	// send post request
	code, body, err := restapi.HTTPPost(url, datasetInfo, "application/json")
	if err != nil || restapi.CheckStatusCode(code) != nil {
		var msg string
		_ = json.Unmarshal(body, &msg)
		fmt.Printf("Failed to create a dataset - code: %d; %s\n", code, msg)
		return nil
	}

	var datasetId string
	err = json.Unmarshal(body, &datasetId)
	if err != nil {
		fmt.Printf("WARNING: Failed to parse resp message: %v", err)
		return nil
	}

	fmt.Println("New dataset created successfully")
	fmt.Printf("\tdataset ID: %s\n", datasetId)

	return nil
}

func Get(params Params) error {
	fmt.Println("Not yet implemented")
	return nil
}

func GetMany(params Params, flagAll bool) error {
	// construct URL
	uriMap := map[string]string{
		"user":  params.User,
		"limit": params.Limit,
	}

	endpoint := restapi.GetDatasetsEndPoint
	if flagAll {
		endpoint = restapi.GetAllDatasetsEndPoint
	}

	url := restapi.CreateURL(params.Endpoint, endpoint, uriMap)

	// send get request
	code, body, err := restapi.HTTPGet(url)
	if err != nil || restapi.CheckStatusCode(code) != nil {
		var msg string
		_ = json.Unmarshal(body, &msg)
		fmt.Printf("Failed to get datasets - code: %d; %s\n", code, msg)
		return nil
	}

	// convert the response into list of struct
	infoList := []openapi.DatasetInfo{}
	err = json.Unmarshal(body, &infoList)
	if err != nil {
		fmt.Printf("Failed to unmarshal dataset info: %v\n", err)
		return nil
	}

	// displaying the output in a table form https://github.com/olekukonko/tablewriter
	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"Dataset ID", "Name", "Description", "Owner", "Public"})
	for _, v := range infoList {
		table.Append([]string{v.Id, v.Name, v.Description, v.UserId, strconv.FormatBool(v.IsPublic)})
	}

	table.Render() // Send output

	return nil
}
