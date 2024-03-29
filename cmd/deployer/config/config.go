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

package config

import (
	"github.com/spf13/afero"
	"gopkg.in/yaml.v3"
)

type Config struct {
	Apiserver string `yaml:"apiserver"`
	Notifier  string `yaml:"notifier"`
	AdminId   string `yaml:"adminId"`
	Region    string `yaml:"region"`
	ComputeId string `yaml:"computeId"`
	Apikey    string `yaml:"apikey"`
	Platform  string `yaml:"platform"`
	Namespace string `yaml:"namespace"`

	JobTemplate JobTemplate `yaml:"jobTemplate"`
}

type JobTemplate struct {
	Folder string `yaml:"folder"`
	File   string `yaml:"file"`
}

var fs afero.Fs

func init() {
	fs = afero.NewOsFs()
}

func LoadConfig(configPath string) (*Config, error) {
	data, err := afero.ReadFile(fs, configPath)
	if err != nil {
		return nil, err
	}

	cfg := &Config{}

	err = yaml.Unmarshal(data, cfg)
	if err != nil {
		return nil, err
	}

	return cfg, nil
}
