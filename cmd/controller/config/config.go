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

package config

import (
	"fmt"

	"github.com/spf13/afero"
	"gopkg.in/yaml.v3"

	"github.com/cisco/fledge/cmd/controller/app/deployer"
	"github.com/cisco/fledge/pkg/util"
)

type Config struct {
	Db       string   `yaml:"db"`
	Brokers  []Broker `yaml:"brokers"`
	Notifier string   `yaml:"notifier"`
	Platform string   `yaml:"platform,omitempty"`
	Port     string   `yaml:"port,omitempty"`
	Registry Registry `yaml:"registry,omitempty"`
}

type Broker struct {
	Sort string `json:"sort" yaml:"sort"`
	Host string `json:"host" yaml:"host"`
}

type Registry struct {
	Sort string `json:"sort" yaml:"sort"`
	Uri  string `json:"uri" yaml:"uri"`
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

	if cfg.Port == "" {
		cfg.Port = fmt.Sprintf("%d", util.ControllerRestApiPort)
	}

	if cfg.Platform == "" {
		cfg.Platform = deployer.K8S
	}

	return cfg, nil
}
