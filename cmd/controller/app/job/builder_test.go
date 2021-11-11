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

package job

import (
	"testing"

	"github.com/cisco/fledge/pkg/openapi"
	"github.com/stretchr/testify/assert"
)

var (
	testJobSpec = openapi.JobSpec{
		UserId:          "testUser",
		Id:              "12345",
		DesignId:        "test",
		SchemaVersion:   "1",
		CodeVersion:     "1",
		Backend:         "mqtt",
		MaxRunTime:      300,
		BaseModelId:     "",
		Hyperparameters: map[string]interface{}{"batchSize": 32, "rounds": 5},
		Dependencies:    []string{"numpy >= 1.2.0"},
	}

	testSchema = openapi.DesignSchema{
		Version: "1",
		Roles: []openapi.Role{
			{Name: "trainer", IsDataConsumer: true},
			{Name: "aggregator"},
			{Name: "gaggr"},
		},
		Channels: []openapi.Channel{
			{
				Name: "param-channel",
				Pair: []string{"aggregator", "trainer"},
				GroupBy: openapi.ChannelGroupBy{
					Type:  "tag",
					Value: []string{"uk", "us"},
				},
			},
			{
				Name: "global-channel",
				Pair: []string{"gaggr", "aggregator"},
			},
		},
	}

	testRoleCode = map[string][]byte{
		"trainer":    []byte("some code1"),
		"aggregator": []byte("some code2"),
		"gaggr":      []byte("some code3"),
	}

	testDatasets = []openapi.DatasetInfo{
		{Url: "https://someurl1.com", Realm: "us|west"},
		{Url: "https://someurl2.com", Realm: "uk"},
	}

	testSchemaWithTwoDataConsumers = openapi.DesignSchema{
		Version: "1",
		Roles: []openapi.Role{
			{Name: "trainer1", IsDataConsumer: true},
			{Name: "trainer2", IsDataConsumer: true},
			{Name: "aggregator1"},
			{Name: "aggregator2"},
		},
		Channels: []openapi.Channel{
			{
				Name: "channel1",
				Pair: []string{"aggregator1", "trainer1"},
				GroupBy: openapi.ChannelGroupBy{
					Type:  "tag",
					Value: []string{"uk", "us"},
				},
			},
			{
				Name: "channel2",
				Pair: []string{"aggregator2", "trainer2"},
				GroupBy: openapi.ChannelGroupBy{
					Type:  "tag",
					Value: []string{"uk", "us"},
				},
			},
			{
				Name: "global-channel",
				Pair: []string{"aggregator1", "aggregator2"},
			},
		},
	}

	testRoleCodeWithTwoDataConsumers = map[string][]byte{
		"trainer1":    []byte("some code1"),
		"trainer2":    []byte("some code2"),
		"aggregator1": []byte("some code3"),
		"aggregator2": []byte("some code4"),
	}
)

func TestGetTaskTemplates(t *testing.T) {
	builder := newJobBuilder(nil, testJobSpec, nil)
	assert.NotNil(t, builder)

	builder.schema = testSchema
	builder.datasets = testDatasets
	builder.roleCode = testRoleCode

	dataRoles, templates := builder.getTaskTemplates()
	assert.NotNil(t, dataRoles)
	assert.Len(t, dataRoles, 1)
	assert.Equal(t, "trainer", dataRoles[0])
	assert.NotNil(t, templates)
	assert.Len(t, templates, 3)
}

func TestPreCheck(t *testing.T) {
	builder := newJobBuilder(nil, testJobSpec, nil)
	assert.NotNil(t, builder)

	builder.schema = testSchema
	builder.datasets = testDatasets

	dataRoles, templates := builder.getTaskTemplates()
	err := builder.preCheck(dataRoles, templates)
	assert.NotNil(t, err)
	assert.Equal(t, "no code found for role trainer", err.Error())

	// set role code
	builder.roleCode = testRoleCode
	err = builder.preCheck(dataRoles, templates)
	assert.Nil(t, err)
}

func TestIsTemplatesConnected(t *testing.T) {
	builder := newJobBuilder(nil, testJobSpec, nil)
	assert.NotNil(t, builder)

	builder.schema = testSchema
	builder.datasets = testDatasets
	builder.roleCode = testRoleCode

	_, templates := builder.getTaskTemplates()
	isConnected := builder.isTemplatesConnected(templates)
	assert.True(t, isConnected)

	savedPair := builder.schema.Channels[0].Pair
	// disconnect one channel
	builder.schema.Channels[0].Pair = []string{}

	_, templates = builder.getTaskTemplates()
	isConnected = builder.isTemplatesConnected(templates)
	assert.False(t, isConnected)
	// restore connection
	builder.schema.Channels[0].Pair = savedPair
}

func TestIsConverging(t *testing.T) {
	builder := newJobBuilder(nil, testJobSpec, nil)
	assert.NotNil(t, builder)

	// success case
	builder.schema = testSchema
	builder.datasets = testDatasets
	builder.roleCode = testRoleCode
	dataRoles, templates := builder.getTaskTemplates()
	res := builder.isConverging(dataRoles, templates)
	assert.True(t, res)

	// failure case
	testSchema.Channels[1].GroupBy.Type = "tag"
	testSchema.Channels[1].GroupBy.Value = []string{"uk", "us"}
	dataRoles, templates = builder.getTaskTemplates()
	res = builder.isConverging(dataRoles, templates)
	assert.False(t, res)
	// reset the changes
	testSchema.Channels[1].GroupBy.Type = ""
	testSchema.Channels[1].GroupBy.Value = nil

	// success case
	builder.schema = testSchemaWithTwoDataConsumers
	builder.datasets = testDatasets
	builder.roleCode = testRoleCodeWithTwoDataConsumers
	dataRoles, templates = builder.getTaskTemplates()
	res = builder.isConverging(dataRoles, templates)
	assert.True(t, res)

	// failure case
	testSchemaWithTwoDataConsumers.Channels[2].GroupBy.Type = "tag"
	testSchemaWithTwoDataConsumers.Channels[2].GroupBy.Value = []string{"uk", "us"}
	dataRoles, templates = builder.getTaskTemplates()
	res = builder.isConverging(dataRoles, templates)
	assert.False(t, res)
	// reset the changes
	testSchemaWithTwoDataConsumers.Channels[2].GroupBy.Type = ""
	testSchemaWithTwoDataConsumers.Channels[2].GroupBy.Value = nil
}

func TestWalk(t *testing.T) {
	builder := newJobBuilder(nil, testJobSpec, nil)
	builder.schema = testSchema
	builder.datasets = testDatasets
	builder.roleCode = testRoleCode

	dataRoles, templates := builder.getTaskTemplates()
	assert.NotNil(t, dataRoles)
	assert.Len(t, dataRoles, 1)
	assert.Equal(t, "trainer", dataRoles[0])
	assert.NotNil(t, templates)
	assert.Len(t, templates, 3)

	// trainer is data role
	tmpl := templates["trainer"]
	tasks, err := tmpl.walk("", templates, builder.datasets)
	assert.Nil(t, err)
	assert.Len(t, tasks, 5)
}
