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

package dataset

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

	DatasetFile string
	DatasetId   string
	Limit       string
}

func Create(params Params) error {
	data, err := ioutil.ReadFile(params.DatasetFile)
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
	code, resp, err := restapi.HTTPPost(url, datasetInfo, "application/json")
	if err != nil || restapi.CheckStatusCode(code) != nil {
		fmt.Printf("Failed to create a dataset - code: %d, error: %v\n", code, err)
		return nil
	}

	var datasetId string
	err = json.Unmarshal(resp, &datasetId)
	if err != nil {
		fmt.Printf("WARNING: Failed to parse resp message: %v", err)
		return nil
	}

	fmt.Println("New dataset created successfully")
	fmt.Printf("\tdataset ID: %s\n", datasetId)

	return nil
}
