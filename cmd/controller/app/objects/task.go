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
	"time"

	"github.com/cisco-open/flame/cmd/controller/config"
	"github.com/cisco-open/flame/pkg/openapi"
	"github.com/cisco-open/flame/pkg/util"
)

type Task struct {
	JobId            string              `json:"jobid"`
	TaskId           string              `json:"taskid"`
	Role             string              `json:"role"`
	IsDataConsumer   bool                `json:"isDataConsumer"`
	Type             openapi.TaskType    `json:"type"`
	Key              string              `json:"key"`
	ComputeId        string              `json:"computeid"`
	Label            []string            `json:"label"`            //label identifies the group associated with the task for the role. Populated during TAG expansion
	ConnectedTaskIds map[string][]string `json:"ConnectedTaskIds"` //tasks can be imagined as nodes. A node can be connected with another node and ConnectedTaskIds store the labels of such nodes.

	// the following are config and code
	JobConfig  JobConfig
	ZippedCode []byte
}

func (t *Task) ToString() string {
	return fmt.Sprintf("Role: %s | "+
		"jobId: %s | "+
		"taskId: %s | "+
		"isDataConsumer: %v | "+
		"label: %s | "+
		"connectedLabels : %s | "+
		"channels: %v",
		t.Role, t.JobId, t.TaskId, t.IsDataConsumer, t.Label, t.ConnectedTaskIds, t.JobConfig.Channels)
}

type JobIdName struct {
	Id   string `json:"id"`
	Name string `json:"name"`
}

type JobConfig struct {
	Job   JobIdName `json:"job"`
	Role  string    `json:"role"`
	Realm string    `json:"realm"`

	Channels []openapi.Channel `json:"channels"`
	Brokers  []config.Broker   `json:"brokers,omitempty"`
	Registry config.Registry   `json:"registry,omitempty"`

	MaxRunTime int32             `json:"maxRunTime,omitempty"`
	ModeSpec   openapi.ModelSpec `json:"modeSpec"`

	//TODO maybe wrap it under DataSpec object?
	DatasetUrl string `json:"dataset,omitempty"`
}

func (tsk *Task) generateTaskId(idx int) {
	h := sha1.New()
	data := fmt.Sprintf("%s-%d-%v-%d", tsk.JobId, idx, tsk.JobConfig, time.Now().UnixNano())
	h.Write([]byte(data))
	tsk.TaskId = fmt.Sprintf("%x", h.Sum(nil))
}

// Configure TODO Update the function name to something else.
func (tsk *Task) Configure(taskType openapi.TaskType, taskKey string, realm string, datasetUrl string, idx int) {
	tsk.Type = taskType
	tsk.Key = taskKey

	tsk.JobConfig.Realm = realm
	tsk.JobConfig.DatasetUrl = datasetUrl

	// generateTaskId() should be called after JobConfig is completely populated
	tsk.generateTaskId(idx)
}

func (cfg *JobConfig) Configure(jobSpec *openapi.JobSpec, brokers []config.Broker, registry config.Registry, role openapi.Role, channels []openapi.Channel) {
	cfg.Job.Id = jobSpec.Id
	cfg.Job.Name = jobSpec.DesignId // DesignId is a string suitable as job's name

	cfg.Role = role.Name
	cfg.Realm = "" // Realm will be updated when datasets are handled

	cfg.MaxRunTime = jobSpec.MaxRunTime
	cfg.ModeSpec = jobSpec.ModelSpec

	cfg.Channels = cfg.extractChannels(role.Name, channels)
	cfg.Brokers = brokers
	cfg.Registry = registry

	cfg.DatasetUrl = "" // Dataset url will be populated when datasets are handled
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
