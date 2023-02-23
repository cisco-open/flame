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

package design

import (
	"github.com/cisco-open/flame/cmd/flamectl/models"
	"github.com/cisco-open/flame/pkg/openapi"
	"github.com/cisco-open/flame/pkg/restapi"
	"go.uber.org/zap"
)

func (c *container) Create(params *models.DesignParams) error {
	logger := c.logger.Sugar().WithOptions(zap.Fields(
		zap.String("user", params.User),
	))

	// construct URL
	uriMap := map[string]string{
		"user": params.User,
	}
	url := restapi.CreateURL(params.Endpoint, restapi.CreateDesignEndPoint, uriMap)

	//Encode the data
	postBody := openapi.DesignInfo{
		Id:          params.Id,
		Description: params.Desc,
	}

	// send post request
	code, responseBody, err := restapi.HTTPPost(url, postBody, "application/json")
	if err != nil || restapi.CheckStatusCode(code) != nil {
		logger.Errorf("Failed to create a new design: code %d\n\n %s\n", code, string(responseBody))
		return err
	}

	logger.Info("New design created successfully")

	return nil
}
