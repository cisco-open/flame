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

package objects

import (
	"crypto/sha1"
	"fmt"

	"github.com/cisco/fledge/cmd/controller/config"
	"github.com/cisco/fledge/pkg/openapi"
)

type Task struct {
	JobId   string `json:"jobid"`
	AgentId string `json:"agentid"`
	Role    string `json:"role"`

	// the following are config and code
	JobConfig  JobConfig
	ZippedCode []byte
}

type JobConfig struct {
	BackEnd  string            `json:"backend"`
	Brokers  []config.Broker   `json:"brokers,omitempty"`
	JobId    string            `json:"jobid"`
	Role     string            `json:"role"`
	Realm    string            `json:"realm"`
	Channels []openapi.Channel `json:"channels"`

	MaxRunTime      int32                  `json:"maxRunTime,omitempty"`
	BaseModelId     string                 `json:"baseModelId,omitempty"`
	Hyperparameters map[string]interface{} `json:"hyperparameters,omitempty"`
	Dependencies    []string               `json:"dependencies,omitempty"`
	DatasetUrl      string                 `json:"dataset,omitempty"`
}

func (p *Task) GenerateAgentId(idx int) {
	h := sha1.New()
	data := fmt.Sprintf("%s-%d-%v", p.JobId, idx, p.JobConfig)
	h.Write([]byte(data))

	p.AgentId = fmt.Sprintf("%x", h.Sum(nil))
}

/*
// For debugging purpose during development
func (jc JobConfig) Print() {
	zap.S().Debug("---")
	zap.S().Debugf("backend: %s\n", jc.BackEnd)
	zap.S().Debugf("broker: %s\n", jc.Broker)
	zap.S().Debugf("JobId: %s\n", jc.JobId)
	zap.S().Debugf("Role: %s\n", jc.Role)
	zap.S().Debugf("Realm: %s\n", jc.Realm)
	for i, channel := range jc.Channels {
		zap.S().Debugf("\t[%d] channel: %v\n", i, channel)
	}

	zap.S().Debugf("MaxRunTime: %d\n", jc.MaxRunTime)
	zap.S().Debugf("BaseModelId: %s\n", jc.BaseModelId)
	zap.S().Debugf("Hyperparameters: %v\n", jc.Hyperparameters)
	zap.S().Debugf("Dependencies: %v\n", jc.Dependencies)
	zap.S().Debugf("DatasetUrl: %s\n", jc.DatasetUrl)
	zap.S().Debug("")
}
*/
