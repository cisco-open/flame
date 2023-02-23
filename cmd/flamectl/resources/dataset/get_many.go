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
	"strconv"

	"github.com/cisco-open/flame/cmd/flamectl/models"
	"github.com/cisco-open/flame/pkg/openapi"
	"github.com/cisco-open/flame/pkg/restapi"
	"github.com/olekukonko/tablewriter"
	"go.uber.org/zap"
)

func (c *container) GetMany(params *models.DatasetParams, flagAll bool) error {
	logger := c.logger.Sugar().WithOptions(zap.Fields(
		zap.String("user", params.User),
		zap.String("limit", params.Limit),
	))

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
		logger.Errorf("Failed to get datasets; code %d\n\n %s", code, string(body))
		return nil
	}

	// convert the response into list of struct
	infoList := []openapi.DatasetInfo{}
	err = json.Unmarshal(body, &infoList)
	if err != nil {
		logger.Errorf("Failed to unmarshal dataset info: %s", err.Error())
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
