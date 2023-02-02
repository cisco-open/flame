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
	Groups   map[string]string `json:"groups"`
	Channels []openapi.Channel `json:"channels"`

	MaxRunTime      int32                  `json:"maxRunTime,omitempty"`
	BaseModel       openapi.BaseModel      `json:"baseModel,omitempty"`
	Hyperparameters map[string]interface{} `json:"hyperparameters,omitempty"`
	Dependencies    []string               `json:"dependencies,omitempty"`
	DatasetUrl      string                 `json:"dataset,omitempty"`
	Optimizer       openapi.Optimizer      `json:"optimizer,omitempty"`
	Selector        openapi.Selector       `json:"selector,omitempty"`
}

func (tsk *Task) generateTaskId(idx int) {
	h := sha1.New()
	data := fmt.Sprintf("%s-%d-%v", tsk.JobId, idx, tsk.JobConfig)
	h.Write([]byte(data))

	tsk.TaskId = fmt.Sprintf("%x", h.Sum(nil))
}

func (tsk *Task) Configure(taskType openapi.TaskType, taskKey string, realm string, datasetUrl string, idx int) {
	tsk.Type = taskType
	tsk.Key = taskKey

	tsk.JobConfig.Realm = realm
	tsk.JobConfig.DatasetUrl = datasetUrl

	// generateTaskId() should be called after JobConfig is completely populated
	tsk.generateTaskId(idx)
}

func (cfg *JobConfig) Configure(jobSpec *openapi.JobSpec, brokers []config.Broker, registry config.Registry,
	role openapi.Role, channels []openapi.Channel) {
	cfg.Job.Id = jobSpec.Id
	// DesignId is a string suitable as job's name
	cfg.Job.Name = jobSpec.DesignId
	cfg.MaxRunTime = jobSpec.MaxRunTime
	cfg.BaseModel = jobSpec.BaseModel
	cfg.Hyperparameters = jobSpec.Hyperparameters
	cfg.Optimizer = jobSpec.Optimizer
	cfg.Selector = jobSpec.Selector
	cfg.Dependencies = jobSpec.Dependencies
	cfg.BackEnd = string(jobSpec.Backend)
	cfg.Brokers = brokers
	cfg.Registry = registry
	// Dataset url will be populated when datasets are handled
	cfg.DatasetUrl = ""

	cfg.Role = role.Name
	// Realm will be updated when datasets are handled
	cfg.Realm = ""
	cfg.Channels = cfg.extractChannels(role.Name, channels)

	// configure the groups of the job based on the groups associated with the assigned role
	cfg.Groups = cfg.extractGroups(role.GroupAssociation)
}

// extractGroups - extracts the associated groups that a given role has of a particular job
func (cfg *JobConfig) extractGroups(groupAssociation []map[string]string) map[string]string {
	groups := make(map[string]string)

	for _, ag := range groupAssociation {
		for key, value := range ag {
			groups[key] = value
		}
	}

	return groups
}

func (cfg *JobConfig) extractChannels(role string, channels []openapi.Channel) []openapi.Channel {
	exChannels := make([]openapi.Channel, 0)

	for _, channel := range channels {
		if util.Contains(channel.Pair, role) {
			exChannels = append(exChannels, channel)
		}
	}

	return exChannels
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
