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

package design

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/olekukonko/tablewriter"
	"go.uber.org/zap"

	"github.com/cisco/fledge/cmd/fledgectl/resources"
	"github.com/cisco/fledge/pkg/openapi"
	"github.com/cisco/fledge/pkg/restapi"
	"github.com/cisco/fledge/pkg/util"
)

type Params struct {
	resources.CommonParams

	DesignId string
	Desc     string
	Limit    string
}

func Create(params Params) error {
	// construct URL
	uriMap := map[string]string{
		"user": params.User,
	}
	url := restapi.CreateURL(params.Endpoint, restapi.CreateDesignEndPoint, uriMap)

	//Encode the data
	postBody := openapi.DesignInfo{
		Id:          params.DesignId,
		Description: params.Desc,
	}

	// send post request
	code, _, err := restapi.HTTPPost(url, postBody, "application/json")
	if err != nil || restapi.CheckStatusCode(code) != nil {
		fmt.Printf("Failed to create a new design - code: %d, error: %v\n", code, err)
		return nil
	}

	fmt.Println("New design created successfully")
	return nil
}

func Get(params Params) error {
	// construct URL
	uriMap := map[string]string{
		"user":     params.User,
		"designId": params.DesignId,
	}
	url := restapi.CreateURL(params.Endpoint, restapi.GetDesignEndPoint, uriMap)

	// send get request
	code, responseBody, err := restapi.HTTPGet(url)
	if err != nil || restapi.CheckStatusCode(code) != nil {
		fmt.Printf("Failed to retrieve design %s - code: %d, error: %v\n", params.DesignId, code, err)
		return nil
	}

	// format the output into prettyJson format
	prettyJSON, err := util.FormatJSON(responseBody)
	if err != nil {
		zap.S().Warnf("error while formating json: %v", err)

		fmt.Println(string(responseBody))
	} else {
		fmt.Println(string(prettyJSON))
	}

	return nil
}

func GetMany(params Params) error {
	// construct URL
	uriMap := map[string]string{
		"user":  params.User,
		"limit": params.Limit,
	}
	url := restapi.CreateURL(params.Endpoint, restapi.GetDesignsEndPoint, uriMap)

	// send get request
	code, responseBody, err := restapi.HTTPGet(url)
	if err != nil || restapi.CheckStatusCode(code) != nil {
		fmt.Printf("Failed to get design templates - code: %d, error: %v\n", code, err)
		return nil
	}

	// convert the response into list of struct
	infoList := []openapi.DesignInfo{}
	err = json.Unmarshal(responseBody, &infoList)
	if err != nil {
		fmt.Printf("Failed to unmarshal design templates: %v\n", err)
		return nil
	}

	// displaying the output in a table form https://github.com/olekukonko/tablewriter
	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"Deisgn ID", "Name", "Description"})

	for _, v := range infoList {
		table.Append([]string{v.Id, v.Name, v.Description})
	}

	table.Render() // Send output

	return nil
}
