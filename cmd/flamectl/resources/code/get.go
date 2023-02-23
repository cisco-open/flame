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

package code

import (
	"os"

	"github.com/cisco-open/flame/cmd/flamectl/models"
	"github.com/cisco-open/flame/pkg/restapi"
	"github.com/cisco-open/flame/pkg/util"
	"go.uber.org/zap"
)

func (c *container) Get(params *models.CodeParams) error {
	logger := c.logger.Sugar().WithOptions(zap.Fields(
		zap.String("user", params.User),
		zap.String("designId", params.DesignId),
		zap.String("version", params.Ver),
	))

	// construct URL
	uriMap := map[string]string{
		"user":     params.User,
		"designId": params.DesignId,
		"version":  params.Ver,
	}
	url := restapi.CreateURL(params.Endpoint, restapi.GetDesignCodeEndPoint, uriMap)

	code, body, err := restapi.HTTPGet(url)
	if err != nil || restapi.CheckStatusCode(code) != nil {
		logger.Errorf("Failed to retrieve a code; code %d\n\n %s", code, string(body))
		return err
	}

	fileName := params.DesignId + "_ver-" + params.Ver + ".zip"
	err = os.WriteFile(fileName, body, util.FilePerm0644)
	if err != nil {
		logger.Errorf("Failed to save file %s: %s", fileName, err.Error())
		return err
	}

	logger.Info("Downloaded %s successfully", fileName)

	return nil
}
