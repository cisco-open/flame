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
	"os"

	"github.com/cisco-open/flame/cmd/flamectl/models"
	"github.com/cisco-open/flame/pkg/openapi"
	"github.com/cisco-open/flame/pkg/restapi"
	"go.uber.org/zap"
)

func (c *container) Create(params *models.DatasetParams) error {
	logger := c.logger.Sugar().WithOptions(zap.Fields(
		zap.String("file", params.File),
	))

	data, err := os.ReadFile(params.File)
	if err != nil {
		logger.Errorf("Failed to read file: %s", err.Error())
		return nil
	}

	// Encode the data
	datasetInfo := openapi.DatasetInfo{}
	err = json.Unmarshal(data, &datasetInfo)
	if err != nil {
		logger.Errorf("Failed to unmarshal: %s", err.Error())
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
		logger.Errorf("Failed to create a dataset; code %d\n\n %s\n", code, string(body))
		return err
	}

	var datasetId string
	err = json.Unmarshal(body, &datasetId)
	if err != nil {
		logger.Errorf("Failed to parse resp message: %s", err.Error())
		return err
	}

	logger.Infof("New dataset ID %s created successfully", datasetId)

	return nil
}
