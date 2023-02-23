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
	"io"
	"net/http"
	"path/filepath"
	"strings"

	"github.com/cisco-open/flame/cmd/flamectl/models"
	"github.com/cisco-open/flame/pkg/restapi"
	"github.com/cisco-open/flame/pkg/util"
	"go.uber.org/zap"
)

func (c *container) Create(params *models.CodeParams) error {
	logger := c.logger.Sugar().WithOptions(zap.Fields(
		zap.String("user", params.User),
		zap.String("designId", params.DesignId),
	))

	// construct URL
	uriMap := map[string]string{
		"user":     params.User,
		"designId": params.DesignId,
	}

	url := restapi.CreateURL(params.Endpoint, restapi.CreateDesignCodeEndPoint, uriMap)

	// "fileName", "fileVer" and "fileData" are names of variables used in openapi specification
	kv := map[string]io.Reader{
		"fileName": strings.NewReader(filepath.Base(params.Path)),
		"fileData": util.MustOpen(params.Path),
	}

	// create multipart/form-data
	buf, writer, err := restapi.CreateMultipartFormData(kv)
	if err != nil {
		logger.Errorf("Failed to create multipart/form-data: %s", err.Error())
		return err
	}

	// send post request
	resp, err := http.Post(url, writer.FormDataContentType(), buf)
	if err != nil {
		body, _ := io.ReadAll(resp.Body)
		logger.Errorf("Failed to create a code: %s\n\n %s", err, string(body))
		return err
	}
	defer resp.Body.Close()

	if err != nil || restapi.CheckStatusCode(resp.StatusCode) != nil {
		body, _ := io.ReadAll(resp.Body)
		logger.Errorf("Failed to create a code - code %d\n\n %s", resp.StatusCode, string(body))
		return err
	}

	logger.Info("Code created successfully for design")

	return nil
}
