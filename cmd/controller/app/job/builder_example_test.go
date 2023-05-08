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

package job

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strconv"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/sv-tools/mongoifc"
	"github.com/tryvium-travels/memongo"
	"github.com/tryvium-travels/memongo/memongolog"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"go.mongodb.org/mongo-driver/mongo/readpref"

	"github.com/cisco-open/flame/cmd/controller/app/database/mongodb"
	"github.com/cisco-open/flame/cmd/controller/app/objects"
	"github.com/cisco-open/flame/cmd/controller/config"
	"github.com/cisco-open/flame/pkg/openapi"
)

const (
	mongoVersion        = "6.0.3"
	testFolder          = "testdata"
	baseExampleFolder   = testFolder + "/examples"
	expectedtasksfolder = testFolder + "/expected_tasks/"
)

var (
	brokers = []config.Broker{
		{
			Sort: string(openapi.MQTT),
			Host: "localhost",
		},
		{
			Sort: string(openapi.P2P),
			Host: "localhost:10104",
		},
	}
)

func Test_examples(t *testing.T) {

	testData := []struct {
		designID            string
		expectedRoles       []string
		datasetNamesByGroup map[string][]string
	}{
		{
			designID:      "asyncfl_hier_mnist",
			expectedRoles: []string{"middle-aggregator", "top-aggregator", "trainer"},
			datasetNamesByGroup: map[string][]string{
				"eu": {
					"dataset_eu_germany.json",
					"dataset_eu_uk.json",
				},
				"na": {
					"dataset_na_canada.json",
					"dataset_na_us.json",
				},
			},
		},
		{
			designID:      "distributed_training",
			expectedRoles: []string{"trainer"},
			datasetNamesByGroup: map[string][]string{
				"us": {
					"dataset_1.json",
					"dataset_2.json",
					"dataset_3.json",
				},
			},
		},
		{
			designID:      "hier_mnist",
			expectedRoles: []string{"middle-aggregator", "top-aggregator", "trainer"},
			datasetNamesByGroup: map[string][]string{
				"eu": {
					"dataset_eu_germany.json",
					"dataset_eu_uk.json",
				},
				"na": {
					"dataset_na_canada.json",
					"dataset_na_us.json",
				},
			},
		},
		{
			designID:      "medmnist",
			expectedRoles: []string{"aggregator", "trainer"},
			datasetNamesByGroup: map[string][]string{
				"us": {
					"dataset1.json",
					"dataset2.json",
					"dataset3.json",
					"dataset4.json",
					"dataset5.json",
					"dataset6.json",
					"dataset7.json",
					"dataset8.json",
					"dataset9.json",
					"dataset10.json",
				},
			},
		},
		{
			designID:      "mnist",
			expectedRoles: []string{"aggregator", "trainer"},
			datasetNamesByGroup: map[string][]string{
				"us": {
					"dataset.json",
				},
			},
		},
		{
			designID:      "parallel_experiment",
			expectedRoles: []string{"aggregator", "trainer"},
			datasetNamesByGroup: map[string][]string{
				"asia": {"dataset_asia_china.json"},
				"uk":   {"dataset_eu_uk.json"},
				"us":   {"dataset_us_west.json"},
			},
		},
		{
			designID:      "hybrid",
			expectedRoles: []string{"aggregator", "trainer"},
			datasetNamesByGroup: map[string][]string{
				"eu": {
					"dataset1.json",
					"dataset2.json",
				},
				"us": {
					"dataset3.json",
					"dataset4.json",
				},
			},
		},
	}

	for _, td := range testData {
		dbService := getDbService(t)
		userID := uuid.NewString()

		t.Run(td.designID, func(t *testing.T) {
			designID := td.designID

			createDesign(t, dbService, userID, designID)
			createDesignSchema(t, dbService, userID, designID)
			createDesignCode(t, dbService, userID, designID)
			datasetGroups := createDatasets(t, dbService, userID, designID, td.datasetNamesByGroup)
			jobSpec := createJob(t, dbService, userID, designID, datasetGroups)

			validateJob(t, dbService, userID, designID, jobSpec, td.expectedRoles)
		})
	}
}

func validateTasks(t *testing.T, testDataPath string, tasks []objects.Task) {
	for i, task := range tasks {
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			var jobConfigData objects.JobConfig
			readFileToStruct(t, fmt.Sprintf("%s/%d.json", testDataPath, i+1), &jobConfigData)

			//assert.Equal(t, jobConfigData, task.JobConfig)
			compareJobConfig(t, jobConfigData, task.JobConfig)
		})
	}
}

func compareJobConfig(t *testing.T, expected, received objects.JobConfig) {
	assert.Equal(t, expected.BackEnd, received.BackEnd)

	assert.Equal(t, expected.Brokers, received.Brokers)
	assert.Equal(t, expected.Registry, received.Registry)
	assert.Equal(t, expected.Job.Name, received.Job.Name)
	assert.Equal(t, expected.Role, received.Role)
	assert.Equal(t, expected.Realm, received.Realm)

	// NOTE: quick and dirty fix
	sort.Slice(expected.Channels, func(i int, j int) bool {
		return expected.Channels[i].Name < expected.Channels[j].Name
	})

	// NOTE: quick and dirty fix
	sort.Slice(received.Channels, func(i int, j int) bool {
		return received.Channels[i].Name < received.Channels[j].Name
	})

	assert.Equal(t, expected.Channels, received.Channels)

	assert.Equal(t, len(expected.GroupAssociation), len(received.GroupAssociation))
	for expectedKey, expectedValue := range expected.GroupAssociation {
		receivedValue, ok := received.GroupAssociation[expectedKey]
		assert.True(t, ok)
		assert.Equal(t, expectedValue, receivedValue)
	}

	assert.Equal(t, expected.MaxRunTime, received.MaxRunTime)
	assert.Equal(t, expected.BaseModel, received.BaseModel)

	assert.Equal(t, len(expected.Hyperparameters), len(received.Hyperparameters))
	for expectedKey, expectedValue := range expected.Hyperparameters {
		receivedValue, ok := received.Hyperparameters[expectedKey]
		assert.True(t, ok)
		assert.Equal(t, expectedValue, receivedValue)
	}

	sort.Strings(expected.Dependencies)
	sort.Strings(received.Dependencies)
	assert.Equal(t, expected.Dependencies, received.Dependencies)

	assert.Equal(t, expected.DatasetUrl, received.DatasetUrl)
	assert.Equal(t, expected.Optimizer, received.Optimizer)
	assert.Equal(t, expected.Selector, received.Selector)
}

func readFileToStruct(t *testing.T, fileName string, i interface{}) {
	content, err := os.ReadFile(fileName)
	assert.NoError(t, err)

	err = json.Unmarshal(content, i)
	assert.NoError(t, err)
}

func connect(t *testing.T) *mongo.Client {
	t.Helper()

	var mongoURI = "mongodb://localhost:27017"

	mongoServer, err := newMongoServer(t)
	if err == nil {
		mongoURI = mongoServer.URI()
	}

	opt := options.Client().ApplyURI(mongoURI)

	cl, err := mongoifc.Connect(context.Background(), opt)
	assert.NoError(t, err)

	t.Cleanup(func() {
		require.NoError(t, cl.Disconnect(context.Background()))
		if mongoServer != nil {
			mongoServer.Stop()
		}
	})

	err = cl.Ping(context.Background(), readpref.Primary())
	require.NoError(t, err)

	client := mongoifc.UnWrapClient(cl)

	return client
}

func newMongoServer(t *testing.T) (*memongo.Server, error) {
	opts := &memongo.Options{
		MongoVersion: mongoVersion,
		LogLevel:     memongolog.LogLevelDebug,
	}

	return memongo.StartWithOptions(opts)
}

func createDesignCode(t *testing.T, dbService *mongodb.MongoService, userID string, designID string) {
	designCodeFile, err := os.Open(fmt.Sprintf("%s/%s/%s.zip", baseExampleFolder, designID, designID))
	assert.NoError(t, err)

	err = dbService.CreateDesignCode(userID, designID, designID, "zip", designCodeFile)
	assert.NoError(t, err)
}

func getDbService(t *testing.T) *mongodb.MongoService {
	ctx := context.Background()
	db := connect(t)

	dbService, err := mongodb.NewMongoServiceWithClient(ctx, db)
	assert.NoError(t, err)

	return dbService
}

func createDesign(t *testing.T, dbService *mongodb.MongoService, userID string, designID string) {
	err := dbService.CreateDesign(userID, openapi.Design{
		Name:        userID,
		UserId:      userID,
		Id:          designID,
		Description: designID + " example",
		Schemas:     []openapi.DesignSchema{},
	})
	assert.NoError(t, err)
}

func createDesignSchema(t *testing.T, dbService *mongodb.MongoService, userID, designID string) {
	schemaFilePath := fmt.Sprintf("%s/%s/schema.json", baseExampleFolder, designID)

	var designSchemaData openapi.DesignSchema
	readFileToStruct(t, schemaFilePath, &designSchemaData)

	err := dbService.CreateDesignSchema(userID, designID, designSchemaData)
	assert.NoError(t, err)
}

func createDataset(t *testing.T, dbService *mongodb.MongoService, userID, designID, fileName string) string {
	datasetFilePath := fmt.Sprintf("%s/%s/%s", baseExampleFolder, designID, fileName)
	var dataset openapi.DatasetInfo
	readFileToStruct(t, datasetFilePath, &dataset)

	id, err := dbService.CreateDataset(userID, dataset)
	assert.NoError(t, err)

	return id
}

func createDatasets(
	t *testing.T,
	dbService *mongodb.MongoService,
	userID, designID string,
	datasetNamesByGroup map[string][]string) map[string][]string {
	m := make(map[string][]string)

	for group, datasetNames := range datasetNamesByGroup {
		for _, datasetName := range datasetNames {
			m[group] = append(m[group], createDataset(t, dbService, userID, designID, datasetName))
		}
	}

	return m
}

func createJob(
	t *testing.T,
	dbService *mongodb.MongoService,
	userID, designID string,
	datasetGroups map[string][]string,
) openapi.JobSpec {
	jobFilePath := fmt.Sprintf("%s/%s/job.json", baseExampleFolder, designID)

	var jobSpecData openapi.JobSpec
	readFileToStruct(t, jobFilePath, &jobSpecData)
	jobSpecData.DataSpec = []openapi.RoleDatasetGroups{
		{
			Role:          "trainer",
			DatasetGroups: datasetGroups,
		},
	}
	jobStatus, err := dbService.CreateJob(userID, jobSpecData)
	assert.NoError(t, err)
	assert.Equal(t, openapi.READY, jobStatus.State)

	jobSpec, err := dbService.GetJob(userID, jobStatus.Id)
	assert.NoError(t, err)

	return jobSpec
}

func validateJob(
	t *testing.T,
	dbService *mongodb.MongoService,
	userID, designID string,
	jobSpec openapi.JobSpec,
	expectedRoles []string,
) {
	builder := NewJobBuilder(dbService, config.JobParams{
		Brokers: brokers,
	})
	assert.NotNil(t, builder)
	tasks, roles, err := builder.GetTasks(&jobSpec)
	assert.NoError(t, err)

	sort.Strings(roles)
	assert.Equal(t, expectedRoles, roles)

	exampleConfigPath := expectedtasksfolder + designID
	validateTasks(t, exampleConfigPath, tasks)
}
