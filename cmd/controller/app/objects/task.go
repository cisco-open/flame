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

package objects

import (
	"crypto/sha1"
	"fmt"
	"sort"

	"github.com/cisco-open/flame/cmd/controller/config"
	"github.com/cisco-open/flame/pkg/openapi"
	"github.com/cisco-open/flame/pkg/util"
)

type Task struct {
	JobId     string           `json:"jobid"`
	TaskId    string           `json:"taskid"`
	Role      string           `json:"role"`
	Type      openapi.TaskType `json:"type"`
	Key       string           `json:"key"`
	ComputeId string           `json:"computeid"`

	// the following are config and code
	JobConfig  JobConfig
	ZippedCode []byte
}

type JobIdName struct {
	Id   string `json:"id"`
	Name string `json:"name"`
}

type JobConfig struct {
	BackEnd  string            `json:"backend"`
	Brokers  []config.Broker   `json:"brokers,omitempty"`
	Registry config.Registry   `json:"registry,omitempty"`
	Job      JobIdName         `json:"job"`
	Role     string            `json:"role"`
	Realm    string            `json:"realm"`
	Channels []openapi.Channel `json:"channels"`
	// Groups is a map of group name to group id
	// structure: channelName:groupName
	GroupAssociation map[string]string `json:"groupAssociation"`

	MaxRunTime      int32                  `json:"maxRunTime,omitempty"`
	BaseModel       openapi.BaseModel      `json:"baseModel,omitempty"`
	Hyperparameters map[string]interface{} `json:"hyperparameters,omitempty"`
	Dependencies    []string               `json:"dependencies,omitempty"`
	DatasetUrl      string                 `json:"dataset,omitempty"`
	Optimizer       openapi.Optimizer      `json:"optimizer,omitempty"`
	Selector        openapi.Selector       `json:"selector,omitempty"`
}

func (tsk *Task) GenerateTaskId(idx int) {
	h := sha1.New()
	data := fmt.Sprintf("%s-%d-%v", tsk.JobId, idx, tsk.JobConfig)
	h.Write([]byte(data))

	tsk.TaskId = fmt.Sprintf("%x", h.Sum(nil))
}

func (cfg *JobConfig) Configure(
	jobSpec *openapi.JobSpec,
	brokers []config.Broker,
	registry config.Registry,
	role openapi.Role,
	channels []openapi.Channel,
) {
	cfg.Job.Id = jobSpec.Id
	// DesignId is a string suitable as job's name
	cfg.Job.Name = jobSpec.DesignId
	cfg.MaxRunTime = jobSpec.MaxRunTime
	cfg.BaseModel = jobSpec.ModelSpec.BaseModel
	cfg.Hyperparameters = jobSpec.ModelSpec.Hyperparameters
	cfg.Optimizer = jobSpec.ModelSpec.Optimizer
	cfg.Selector = jobSpec.ModelSpec.Selector
	cfg.Dependencies = jobSpec.ModelSpec.Dependencies
	cfg.BackEnd = string(jobSpec.Backend)
	cfg.Brokers = brokers
	cfg.Registry = registry
	// Dataset url will be populated when datasets are handled
	cfg.DatasetUrl = ""
	cfg.Channels = cfg.extractChannels(role.Name, channels)

	cfg.Role = role.Name
}

func (cfg *JobConfig) extractChannels(role string, channels []openapi.Channel) []openapi.Channel {
	exChannels := make([]openapi.Channel, 0)

	for _, channel := range channels {
		if util.Contains(channel.Pair, role) {
			exChannels = append(exChannels, channel)
		}
	}

	sort.Slice(exChannels, func(i, j int) bool {
		return channels[i].Name < channels[j].Name
	})

	return exChannels
}
