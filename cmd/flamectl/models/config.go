// Copyright 2023 Cisco Systems, Inc. and its affiliates
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

package models

// Configuration holds the details for connecting to the api-server
type Configuration struct {
	ApiServer ApiServer `yaml:"apiserver"`
	User      string    `yaml:"user"`
}

// ApiServer holds the parameter details for connecting to an api-server
type ApiServer struct {
	// Endpoint holds the address of the api-server
	Endpoint string `yaml:"endpoint"`
}

// NewConfiguration creates a new configuration instance
func NewConfiguration() *Configuration {
	return new(Configuration)
}
