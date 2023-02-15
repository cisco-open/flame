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

package schema

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/cisco-open/flame/cmd/flamectl/resources"
	"github.com/cisco-open/flame/pkg/openapi"
	"github.com/cisco-open/flame/pkg/openapi/constants"
	"github.com/cisco-open/flame/pkg/restapi"
	"github.com/cisco-open/flame/pkg/util"
)

type Params struct {
	resources.CommonParams

	DesignId   string
	SchemaPath string
	Version    string
}

func Create(params Params) error {
	// read schema via conf file
	jsonData, err := os.ReadFile(params.SchemaPath)
	if err != nil {
		return err
	}

	// read schemas from the json file
	schema := openapi.DesignSchema{}
	err = json.Unmarshal(jsonData, &schema)
	if err != nil {
		return err
	}

	// construct URL
	uriMap := map[string]string{
		constants.ParamUser:     params.User,
		constants.ParamDesignID: params.DesignId,
	}
	url := restapi.CreateURL(params.Endpoint, restapi.CreateDesignSchemaEndPoint, uriMap)

	// send post request
	code, responseBody, err := restapi.HTTPPost(url, schema, "application/json")
	if err != nil || restapi.CheckStatusCode(code) != nil {
		var msg string
		_ = json.Unmarshal(responseBody, &msg)
		fmt.Printf("Failed to create a new schema - code: %d; %s\n", code, msg)
		return nil
	}

	fmt.Printf("Schema created successfully for design '%s'\n", params.DesignId)

	return nil
}

func Get(params Params) error {
	// construct URL
	uriMap := map[string]string{
		constants.ParamUser:     params.User,
		constants.ParamDesignID: params.DesignId,
		constants.ParamVersion:  params.Version,
	}
	url := restapi.CreateURL(params.Endpoint, restapi.GetDesignSchemaEndPoint, uriMap)

	// send get request
	code, responseBody, err := restapi.HTTPGet(url)
	if err != nil || restapi.CheckStatusCode(code) != nil {
		var msg string
		_ = json.Unmarshal(responseBody, &msg)
		fmt.Printf("Failed to get a schema of version '%s' for design '%s' - code: %d; %s\n",
			params.Version, params.DesignId, code, msg)
		return nil
	}

	// format the output into prettyJson format
	prettyJSON, err := util.FormatJSON(responseBody)
	if err != nil {
		fmt.Printf("WARNING: error while formating json: %v\n\n", err)

		fmt.Println(string(responseBody))
	} else {
		fmt.Println(string(prettyJSON))
	}

	return nil
}

func GetMany(params Params) error {
	// construct URL
	uriMap := map[string]string{
		constants.ParamUser:     params.User,
		constants.ParamDesignID: params.DesignId,
	}
	url := restapi.CreateURL(params.Endpoint, restapi.GetDesignSchemasEndPoint, uriMap)

	// send get request
	code, responseBody, err := restapi.HTTPGet(url)
	if err != nil || restapi.CheckStatusCode(code) != nil {
		var msg string
		_ = json.Unmarshal(responseBody, &msg)
		fmt.Printf("Failed to get design schemas for design '%s' - code: %d; %s\n", params.DesignId, code, msg)
		return nil
	}

	// format the output into prettyJson format
	prettyJSON, err := util.FormatJSON(responseBody)
	if err != nil {
		fmt.Printf("WARNING: error while formating json: %v\n\n", err)

		fmt.Println(string(responseBody))
	} else {
		fmt.Println(string(prettyJSON))
	}

	return nil
}

func Update(params Params) error {
	// read schema via conf file
	jsonData, err := os.ReadFile(params.SchemaPath)
	if err != nil {
		return err
	}

	// read schema from the json file
	schema := openapi.DesignSchema{}
	err = json.Unmarshal(jsonData, &schema)
	if err != nil {
		return err
	}

	// construct URL
	uriMap := map[string]string{
		constants.ParamUser:     params.User,
		constants.ParamDesignID: params.DesignId,
		constants.ParamVersion:  params.Version,
	}
	url := restapi.CreateURL(params.Endpoint, restapi.UpdateDesignSchemaEndPoint, uriMap)

	code, responseBody, err := restapi.HTTPPut(url, schema, "application/json")
	if err != nil && restapi.CheckStatusCode(code) != nil {
		var msg string
		_ = json.Unmarshal(responseBody, &msg)
		fmt.Printf("Failed to update schema version '%s' - code: %d; %s\n", params.Version, code, msg)
		return nil
	}

	fmt.Printf("Updated schema version '%s' successfully for design %s\n", params.Version, params.DesignId)

	return nil
}

func Remove(params Params) error {
	// construct URL
	uriMap := map[string]string{
		constants.ParamUser:     params.User,
		constants.ParamDesignID: params.DesignId,
		constants.ParamVersion:  params.Version,
	}
	url := restapi.CreateURL(params.Endpoint, restapi.DeleteDesignSchemaEndPoint, uriMap)

	statusCode, responseBody, err := restapi.HTTPDelete(url, nil, "")
	if err != nil || restapi.CheckStatusCode(statusCode) != nil {
		var msg string
		_ = json.Unmarshal(responseBody, &msg)
		fmt.Printf("Failed to delete schema of version '%s' - code: %d; %s\n",
			params.Version, statusCode, msg)
		return nil
	}

	fmt.Printf("Deleted schema version '%s' successfully\n", params.Version)

	return nil
}
