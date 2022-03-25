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

package schema

import (
	"encoding/json"
	"fmt"
	"io/ioutil"

	"github.com/cisco-open/flame/cmd/flamectl/resources"
	"github.com/cisco-open/flame/pkg/openapi"
	"github.com/cisco-open/flame/pkg/restapi"
	"github.com/cisco-open/flame/pkg/util"
	"github.com/prometheus/common/log"
)

type Params struct {
	resources.CommonParams

	DesignId   string
	SchemaPath string
	Version    string
}

func Create(params Params) error {
	// read schema via conf file
	jsonData, err := ioutil.ReadFile(params.SchemaPath)
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
		"user":     params.User,
		"designId": params.DesignId,
	}
	url := restapi.CreateURL(params.Endpoint, restapi.CreateDesignSchemaEndPoint, uriMap)

	// send post request
	code, _, err := restapi.HTTPPost(url, schema, "application/json")
	if err != nil || restapi.CheckStatusCode(code) != nil {
		fmt.Printf("Failed to create a new schema - code: %d, error: %v\n", code, err)
		return nil
	}

	fmt.Printf("Schema created successfully for design %s\n", params.DesignId)

	return nil
}

func Get(params Params) error {
	// construct URL
	uriMap := map[string]string{
		"user":     params.User,
		"designId": params.DesignId,
		"version":  params.Version,
	}
	url := restapi.CreateURL(params.Endpoint, restapi.GetDesignSchemaEndPoint, uriMap)

	// send get request
	code, responseBody, err := restapi.HTTPGet(url)
	if err != nil || restapi.CheckStatusCode(code) != nil {
		fmt.Printf("Failed to get a design schema of version %s for design %s - code: %d, error: %v\n",
			params.Version, params.DesignId, code, err)
		return nil
	}

	// format the output into prettyJson format
	prettyJSON, err := util.FormatJSON(responseBody)
	if err != nil {
		log.Warnf("error while formating json: %v", err)

		fmt.Println(string(responseBody))
	} else {
		fmt.Println(string(prettyJSON))
	}

	return nil
}

func GetMany(params Params) error {
	// construct URL
	uriMap := map[string]string{
		"user":     params.User,
		"designId": params.DesignId,
	}
	url := restapi.CreateURL(params.Endpoint, restapi.GetDesignSchemasEndPoint, uriMap)

	// send get request
	code, responseBody, err := restapi.HTTPGet(url)
	if err != nil || restapi.CheckStatusCode(code) != nil {
		fmt.Printf("Failed to get design schemas for design %s - code: %d, error: %v\n", params.DesignId, code, err)
		return nil
	}

	// format the output into prettyJson format
	prettyJSON, err := util.FormatJSON(responseBody)
	if err != nil {
		log.Warnf("error while formating json: %v", err)

		fmt.Println(string(responseBody))
	} else {
		fmt.Println(string(prettyJSON))
	}

	return nil
}

func Update(params Params) error {
	// read schema via conf file
	jsonData, err := ioutil.ReadFile(params.SchemaPath)
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
		"user":     params.User,
		"designId": params.DesignId,
		"version":  params.Version,
	}
	url := restapi.CreateURL(params.Endpoint, restapi.UpdateDesignSchemaEndPoint, uriMap)

	code, _, err := restapi.HTTPPut(url, schema, "application/json")
	if err != nil && restapi.CheckStatusCode(code) != nil {
		fmt.Printf("error while updating a schema version %s - code: %d, error: %v\n", params.Version, code, err)
		return nil
	}

	fmt.Printf("Updated schema version %s successfully for design %s\n", params.Version, params.DesignId)

	return nil
}
