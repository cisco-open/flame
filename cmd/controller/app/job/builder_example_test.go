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
	mongoVersion      = "6.0.3"
	rootFolder        = "../../../../"
	baseExampleFolder = rootFolder + "/examples"
)

func Test_asyncfl_hier_mnist(t *testing.T) {
	rootExample := baseExampleFolder + "/asyncfl_hier_mnist"

	designCodeFile, err := os.Open(rootExample + "/asyncfl_hier_mnist.zip")
	assert.NoError(t, err)

	db := connect(t)
	userID := uuid.NewString()
	ctx := context.Background()

	dbService, err := mongodb.NewMongoServiceWithClient(ctx, db)
	assert.NoError(t, err)

	builder := NewJobBuilder(dbService, config.JobParams{})
	assert.NotNil(t, builder)

	designID := "asyncfl_hier_mnist"

	err = dbService.CreateDesign(userID, openapi.Design{
		Name:        userID,
		UserId:      userID,
		Id:          designID,
		Description: "asynchronous hierarchical FL mnist example",
		Schemas:     []openapi.DesignSchema{},
	})
	assert.NoError(t, err)

	var designSchemaData openapi.DesignSchema
	readFileToStruct(t, rootExample+"/schema.json", &designSchemaData)
	err = dbService.CreateDesignSchema(userID, designID, designSchemaData)
	assert.NoError(t, err)

	err = dbService.CreateDesignCode(userID, designID, "asyncfl_hier_mnist", "zip", designCodeFile)
	assert.NoError(t, err)

	var datasetEuGermany, datasetEuUk, datasetNaCanada, datasetNaUs openapi.DatasetInfo
	readFileToStruct(t, rootExample+"/dataset_eu_germany.json", &datasetEuGermany)
	readFileToStruct(t, rootExample+"/dataset_eu_uk.json", &datasetEuUk)
	readFileToStruct(t, rootExample+"/dataset_na_canada.json", &datasetNaCanada)
	readFileToStruct(t, rootExample+"/dataset_na_us.json", &datasetNaUs)
	datasetEuGermanyID, err := dbService.CreateDataset(userID, datasetEuGermany)
	assert.NoError(t, err)
	datasetEuUkID, err := dbService.CreateDataset(userID, datasetEuUk)
	assert.NoError(t, err)
	datasetNaCanadaID, err := dbService.CreateDataset(userID, datasetNaCanada)
	assert.NoError(t, err)
	datasetNaUsID, err := dbService.CreateDataset(userID, datasetNaUs)
	assert.NoError(t, err)

	var jobSpecData openapi.JobSpec
	readFileToStruct(t, rootExample+"/job.json", &jobSpecData)
	jobSpecData.DataSpec.FromSystem = map[string]map[string][]string{
		"trainer": {
			"eu": []string{datasetEuGermanyID, datasetEuUkID},
			"na": []string{datasetNaCanadaID, datasetNaUsID},
		},
	}
	jobStatus, err := dbService.CreateJob(userID, jobSpecData)
	assert.NoError(t, err)
	assert.Equal(t, openapi.READY, jobStatus.State)

	jobSpec, err := dbService.GetJob(userID, jobStatus.Id)
	assert.NoError(t, err)

	tasks, roles, err := builder.GetTasks(&jobSpec)
	assert.NoError(t, err)

	sort.Strings(roles)
	expectedRoles := []string{"middle-aggregator", "top-aggregator", "trainer"}
	assert.Equal(t, expectedRoles, roles)

	exampleConfigPath := "testdata/" + designID
	validateTasks(t, exampleConfigPath, tasks)
}

func Test_distributed_training(t *testing.T) {
	rootExample := baseExampleFolder + "/distributed_training"

	db := connect(t)
	userID := uuid.NewString()
	ctx := context.Background()

	dbService, err := mongodb.NewMongoServiceWithClient(ctx, db)
	assert.NoError(t, err)

	builder := NewJobBuilder(dbService, config.JobParams{})
	assert.NotNil(t, builder)

	designID := "distributed_training"

	err = dbService.CreateDesign(userID, openapi.Design{
		Name:        userID,
		UserId:      userID,
		Id:          designID,
		Description: "distributed training",
		Schemas:     []openapi.DesignSchema{},
	})
	assert.NoError(t, err)

	var designSchemaData openapi.DesignSchema
	readFileToStruct(t, rootExample+"/schema.json", &designSchemaData)
	err = dbService.CreateDesignSchema(userID, designID, designSchemaData)
	assert.NoError(t, err)

	designCodeFile, err := os.Open(rootExample + "/distributed_training.zip")
	assert.NoError(t, err)
	err = dbService.CreateDesignCode(userID, designID, "distributed_training", "zip", designCodeFile)
	assert.NoError(t, err)

	var dataset1, dataset2, dataset3 openapi.DatasetInfo
	readFileToStruct(t, rootExample+"/dataset_1.json", &dataset1)
	readFileToStruct(t, rootExample+"/dataset_2.json", &dataset2)
	readFileToStruct(t, rootExample+"/dataset_3.json", &dataset3)
	dataset1ID, err := dbService.CreateDataset(userID, dataset1)
	assert.NoError(t, err)
	dataset2ID, err := dbService.CreateDataset(userID, dataset2)
	assert.NoError(t, err)
	dataset3ID, err := dbService.CreateDataset(userID, dataset3)
	assert.NoError(t, err)

	var jobSpecData openapi.JobSpec
	readFileToStruct(t, rootExample+"/job.json", &jobSpecData)
	jobSpecData.DataSpec.FromSystem = map[string]map[string][]string{
		"trainer": {
			"us": []string{dataset1ID, dataset2ID, dataset3ID},
		},
	}
	jobStatus, err := dbService.CreateJob(userID, jobSpecData)
	assert.NoError(t, err)
	assert.Equal(t, openapi.READY, jobStatus.State)

	jobSpec, err := dbService.GetJob(userID, jobStatus.Id)
	assert.NoError(t, err)

	tasks, roles, err := builder.GetTasks(&jobSpec)
	assert.NoError(t, err)

	sort.Strings(roles)
	expectedRoles := []string{"trainer"}
	assert.Equal(t, expectedRoles, roles)

	exampleConfigPath := "testdata/" + designID
	validateTasks(t, exampleConfigPath, tasks)
}

func Test_hier_mnist(t *testing.T) {
	rootExample := baseExampleFolder + "/hier_mnist"

	db := connect(t)
	userID := uuid.NewString()
	ctx := context.Background()

	dbService, err := mongodb.NewMongoServiceWithClient(ctx, db)
	assert.NoError(t, err)

	builder := NewJobBuilder(dbService, config.JobParams{})
	assert.NotNil(t, builder)

	designID := "hier_mnist"

	err = dbService.CreateDesign(userID, openapi.Design{
		Name:        userID,
		UserId:      userID,
		Id:          designID,
		Description: "hierarchical FL mnist example",
		Schemas:     []openapi.DesignSchema{},
	})
	assert.NoError(t, err)

	var designSchemaData openapi.DesignSchema
	readFileToStruct(t, rootExample+"/schema.json", &designSchemaData)
	err = dbService.CreateDesignSchema(userID, designID, designSchemaData)
	assert.NoError(t, err)

	designCodeFile, err := os.Open(rootExample + "/hier_mnist.zip")
	assert.NoError(t, err)
	err = dbService.CreateDesignCode(userID, designID, "hier_mnist", "zip", designCodeFile)
	assert.NoError(t, err)

	var datasetEuGermany, datasetEuUk, datasetNaCanada, datasetNaUs openapi.DatasetInfo
	readFileToStruct(t, rootExample+"/dataset_eu_germany.json", &datasetEuGermany)
	readFileToStruct(t, rootExample+"/dataset_eu_uk.json", &datasetEuUk)
	readFileToStruct(t, rootExample+"/dataset_na_canada.json", &datasetNaCanada)
	readFileToStruct(t, rootExample+"/dataset_na_us.json", &datasetNaUs)
	datasetEuGermanyID, err := dbService.CreateDataset(userID, datasetEuGermany)
	assert.NoError(t, err)
	datasetEuUkID, err := dbService.CreateDataset(userID, datasetEuUk)
	assert.NoError(t, err)
	datasetNaCanadaID, err := dbService.CreateDataset(userID, datasetNaCanada)
	assert.NoError(t, err)
	datasetNaUsID, err := dbService.CreateDataset(userID, datasetNaUs)
	assert.NoError(t, err)

	var jobSpecData openapi.JobSpec
	readFileToStruct(t, rootExample+"/job.json", &jobSpecData)
	jobSpecData.DataSpec.FromSystem = map[string]map[string][]string{
		"trainer": {
			"eu": []string{datasetEuGermanyID, datasetEuUkID},
			"na": []string{datasetNaCanadaID, datasetNaUsID},
		},
	}
	jobStatus, err := dbService.CreateJob(userID, jobSpecData)
	assert.NoError(t, err)
	assert.Equal(t, openapi.READY, jobStatus.State)

	jobSpec, err := dbService.GetJob(userID, jobStatus.Id)
	assert.NoError(t, err)

	tasks, roles, err := builder.GetTasks(&jobSpec)
	assert.NoError(t, err)

	sort.Strings(roles)
	expectedRoles := []string{"middle-aggregator", "top-aggregator", "trainer"}
	assert.Equal(t, expectedRoles, roles)

	exampleConfigPath := "testdata/" + designID
	validateTasks(t, exampleConfigPath, tasks)
}

func Test_medmnist(t *testing.T) {
	rootExample := baseExampleFolder + "/medmnist"

	db := connect(t)
	userID := uuid.NewString()
	ctx := context.Background()

	dbService, err := mongodb.NewMongoServiceWithClient(ctx, db)
	assert.NoError(t, err)

	builder := NewJobBuilder(dbService, config.JobParams{})
	assert.NotNil(t, builder)

	designID := "medmnist"

	err = dbService.CreateDesign(userID, openapi.Design{
		Name:        userID,
		UserId:      userID,
		Id:          designID,
		Description: "MedMNIST",
		Schemas:     []openapi.DesignSchema{},
	})
	assert.NoError(t, err)

	var designSchemaData openapi.DesignSchema
	readFileToStruct(t, rootExample+"/schema.json", &designSchemaData)
	err = dbService.CreateDesignSchema(userID, designID, designSchemaData)
	assert.NoError(t, err)

	designCodeFile, err := os.Open(rootExample + "/medmnist.zip")
	assert.NoError(t, err)
	err = dbService.CreateDesignCode(userID, designID, "medmnist", "zip", designCodeFile)
	assert.NoError(t, err)

	var dataset1, dataset2, dataset3, dataset4, dataset5,
		dataset6, dataset7, dataset8, dataset9, dataset10 openapi.DatasetInfo
	readFileToStruct(t, rootExample+"/dataset1.json", &dataset1)
	readFileToStruct(t, rootExample+"/dataset2.json", &dataset2)
	readFileToStruct(t, rootExample+"/dataset3.json", &dataset3)
	readFileToStruct(t, rootExample+"/dataset4.json", &dataset4)
	readFileToStruct(t, rootExample+"/dataset5.json", &dataset5)
	readFileToStruct(t, rootExample+"/dataset6.json", &dataset6)
	readFileToStruct(t, rootExample+"/dataset7.json", &dataset7)
	readFileToStruct(t, rootExample+"/dataset8.json", &dataset8)
	readFileToStruct(t, rootExample+"/dataset9.json", &dataset9)
	readFileToStruct(t, rootExample+"/dataset10.json", &dataset10)
	dataset1ID, err := dbService.CreateDataset(userID, dataset1)
	assert.NoError(t, err)
	dataset2ID, err := dbService.CreateDataset(userID, dataset2)
	assert.NoError(t, err)
	dataset3ID, err := dbService.CreateDataset(userID, dataset3)
	assert.NoError(t, err)
	dataset4ID, err := dbService.CreateDataset(userID, dataset4)
	assert.NoError(t, err)
	dataset5ID, err := dbService.CreateDataset(userID, dataset5)
	assert.NoError(t, err)
	dataset6ID, err := dbService.CreateDataset(userID, dataset6)
	assert.NoError(t, err)
	dataset7ID, err := dbService.CreateDataset(userID, dataset7)
	assert.NoError(t, err)
	dataset8ID, err := dbService.CreateDataset(userID, dataset8)
	assert.NoError(t, err)
	dataset9ID, err := dbService.CreateDataset(userID, dataset9)
	assert.NoError(t, err)
	dataset10ID, err := dbService.CreateDataset(userID, dataset10)
	assert.NoError(t, err)

	var jobSpecData openapi.JobSpec
	readFileToStruct(t, rootExample+"/job.json", &jobSpecData)
	jobSpecData.DataSpec.FromSystem = map[string]map[string][]string{
		"trainer": {
			"us": []string{
				dataset1ID, dataset2ID, dataset3ID, dataset4ID, dataset5ID,
				dataset6ID, dataset7ID, dataset8ID, dataset9ID, dataset10ID,
			},
		},
	}
	jobStatus, err := dbService.CreateJob(userID, jobSpecData)
	assert.NoError(t, err)
	assert.Equal(t, openapi.READY, jobStatus.State)

	jobSpec, err := dbService.GetJob(userID, jobStatus.Id)
	assert.NoError(t, err)

	tasks, roles, err := builder.GetTasks(&jobSpec)
	assert.NoError(t, err)

	sort.Strings(roles)
	expectedRoles := []string{"aggregator", "trainer"}
	assert.Equal(t, expectedRoles, roles)

	exampleConfigPath := "testdata/" + designID
	validateTasks(t, exampleConfigPath, tasks)
}

func Test_mnist(t *testing.T) {
	rootExample := baseExampleFolder + "/mnist"

	db := connect(t)
	userID := uuid.NewString()
	ctx := context.Background()

	dbService, err := mongodb.NewMongoServiceWithClient(ctx, db)
	assert.NoError(t, err)

	builder := NewJobBuilder(dbService, config.JobParams{})
	assert.NotNil(t, builder)

	designID := "mnist"

	err = dbService.CreateDesign(userID, openapi.Design{
		Name:        userID,
		UserId:      userID,
		Id:          designID,
		Description: "mnist example",
		Schemas:     []openapi.DesignSchema{},
	})
	assert.NoError(t, err)

	var designSchemaData openapi.DesignSchema
	readFileToStruct(t, rootExample+"/schema.json", &designSchemaData)
	err = dbService.CreateDesignSchema(userID, designID, designSchemaData)
	assert.NoError(t, err)

	designCodeFile, err := os.Open(rootExample + "/mnist.zip")
	assert.NoError(t, err)
	err = dbService.CreateDesignCode(userID, designID, "mnist", "zip", designCodeFile)
	assert.NoError(t, err)

	var dataset openapi.DatasetInfo
	readFileToStruct(t, rootExample+"/dataset.json", &dataset)
	datasetID, err := dbService.CreateDataset(userID, dataset)
	assert.NoError(t, err)

	var jobSpecData openapi.JobSpec
	readFileToStruct(t, rootExample+"/job.json", &jobSpecData)
	jobSpecData.DataSpec.FromSystem = map[string]map[string][]string{
		"trainer": {
			"us": []string{datasetID},
		},
	}
	jobStatus, err := dbService.CreateJob(userID, jobSpecData)
	assert.NoError(t, err)
	assert.Equal(t, openapi.READY, jobStatus.State)

	jobSpec, err := dbService.GetJob(userID, jobStatus.Id)
	assert.NoError(t, err)

	tasks, roles, err := builder.GetTasks(&jobSpec)
	assert.NoError(t, err)

	sort.Strings(roles)
	expectedRoles := []string{"aggregator", "trainer"}
	assert.Equal(t, expectedRoles, roles)

	exampleConfigPath := "testdata/" + designID
	validateTasks(t, exampleConfigPath, tasks)
}
func Test_parallel_experiment(t *testing.T) {
	rootExample := baseExampleFolder + "/parallel_experiment"

	db := connect(t)
	userID := uuid.NewString()
	ctx := context.Background()

	dbService, err := mongodb.NewMongoServiceWithClient(ctx, db)
	assert.NoError(t, err)

	builder := NewJobBuilder(dbService, config.JobParams{})
	assert.NotNil(t, builder)

	designID := "parallel_experiment"

	err = dbService.CreateDesign(userID, openapi.Design{
		Name:        userID,
		UserId:      userID,
		Id:          designID,
		Description: "parallel exp",
		Schemas:     []openapi.DesignSchema{},
	})
	assert.NoError(t, err)

	var designSchemaData openapi.DesignSchema
	readFileToStruct(t, rootExample+"/schema.json", &designSchemaData)
	err = dbService.CreateDesignSchema(userID, designID, designSchemaData)
	assert.NoError(t, err)

	designCodeFile, err := os.Open(rootExample + "/parallel_experiment.zip")
	assert.NoError(t, err)
	err = dbService.CreateDesignCode(userID, designID, "parallel_experiment", "zip", designCodeFile)
	assert.NoError(t, err)

	var datasetAsia, datasetEuUk, datasetUsWest openapi.DatasetInfo
	readFileToStruct(t, rootExample+"/dataset_asia_china.json", &datasetAsia)
	readFileToStruct(t, rootExample+"/dataset_eu_uk.json", &datasetEuUk)
	readFileToStruct(t, rootExample+"/dataset_us_west.json", &datasetUsWest)
	datasetAsiaID, err := dbService.CreateDataset(userID, datasetAsia)
	assert.NoError(t, err)
	datasetEuUkID, err := dbService.CreateDataset(userID, datasetEuUk)
	assert.NoError(t, err)
	datasetUsWestID, err := dbService.CreateDataset(userID, datasetUsWest)
	assert.NoError(t, err)

	var jobSpecData openapi.JobSpec
	readFileToStruct(t, rootExample+"/job.json", &jobSpecData)
	jobSpecData.DataSpec.FromSystem = map[string]map[string][]string{
		"trainer": {
			"asia": []string{datasetAsiaID},
			"uk":   []string{datasetEuUkID},
			"us":   []string{datasetUsWestID},
		},
	}
	jobStatus, err := dbService.CreateJob(userID, jobSpecData)
	assert.NoError(t, err)
	assert.Equal(t, openapi.READY, jobStatus.State)

	jobSpec, err := dbService.GetJob(userID, jobStatus.Id)
	assert.NoError(t, err)

	tasks, roles, err := builder.GetTasks(&jobSpec)
	assert.NoError(t, err)

	sort.Strings(roles)
	expectedRoles := []string{"aggregator", "trainer"}
	assert.Equal(t, expectedRoles, roles)

	exampleConfigPath := "testdata/" + designID
	validateTasks(t, exampleConfigPath, tasks)
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
