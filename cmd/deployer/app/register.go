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

package app

import (
	"crypto/tls"
	"net/http"
	"os"

	"go.uber.org/zap"

	"github.com/cisco-open/flame/pkg/openapi"
	"github.com/cisco-open/flame/pkg/restapi"
)

type ComputeResource struct {
	apiserverEp string
	name        string
	spec        openapi.ComputeSpec
	registered  bool
}

func NewCompute(apiserverEp string, computeSpec openapi.ComputeSpec, bInsecure bool, bPlain bool) (*ComputeResource, error) {
	name, err := os.Hostname()
	if err != nil {
		return nil, err
	}

	if bPlain {
	} else {
		tlsCfg := &tls.Config{}
		if bInsecure {
			zap.S().Warn("Warning: allow insecure connection\n")

			tlsCfg.InsecureSkipVerify = true
			http.DefaultTransport.(*http.Transport).TLSClientConfig = tlsCfg
		}
	}

	compute := &ComputeResource{
		apiserverEp: apiserverEp,
		name:        name,
		spec:        computeSpec,
		registered:  false,
	}

	return compute, nil
}

func (compute *ComputeResource) RegisterNewCompute() error {
	// construct URL
	uriMap := map[string]string{}
	url := restapi.CreateURL(compute.apiserverEp, restapi.RegisterComputeEndpoint, uriMap)

	code, resp, err := restapi.HTTPPost(url, compute.spec, "application/json")
	if err != nil || restapi.CheckStatusCode(code) != nil {
		zap.S().Errorf("Failed to register compute, sent computeSpec: %v, resp: %v, code: %d, err: %v",
			compute.spec, string(resp), code, err)
		return err
	}

	zap.S().Infof("Success in registering new compute, sent obj: %v, resp: %v, code: %d",
		compute.spec, string(resp), code)
	compute.registered = true

	return nil
}
