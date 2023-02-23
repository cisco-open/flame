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
	"os"

	"github.com/cisco-open/flame/cmd/flamectl/models"
	"github.com/cisco-open/flame/pkg/openapi"
	"github.com/cisco-open/flame/pkg/restapi"
	"go.uber.org/zap"
)

func (c *container) Update(params *models.SchemaParams) error {
	logger := c.logger.Sugar().WithOptions(zap.Fields(
		zap.String("user", params.User),
		zap.String("designId", params.DesignId),
		zap.String("version", params.Version),
	))

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
		"user":     params.User,
		"designId": params.DesignId,
		"version":  params.Version,
	}
	url := restapi.CreateURL(params.Endpoint, restapi.UpdateDesignSchemaEndPoint, uriMap)

	code, responseBody, err := restapi.HTTPPut(url, schema, "application/json")
	if err != nil && restapi.CheckStatusCode(code) != nil {
		var msg string
		_ = json.Unmarshal(responseBody, &msg)
		logger.Errorf("Failed to update schema; code %d\n\n %s\n\n %s", code, err.Error(), string(responseBody))
		return nil
	}

	logger.Info("Updated schema successfully")

	return nil
}
