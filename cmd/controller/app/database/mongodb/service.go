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

package mongodb

import (
	"context"
	"time"

	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"go.mongodb.org/mongo-driver/mongo/readpref"
	"go.mongodb.org/mongo-driver/x/bsonx"
	"go.uber.org/zap"
)

const (
	DatabaseName      = "flame"
	DatasetCollection = "t_dataset"
	DesignCollection  = "t_design"
	JobCollection     = "t_job"
	TaskCollection    = "t_task"

	orderAscend  int32 = 1
	orderDescend int32 = -1
)

type MongoService struct {
	client            *mongo.Client
	database          *mongo.Database
	datasetCollection *mongo.Collection
	designCollection  *mongo.Collection
	jobCollection     *mongo.Collection
	taskCollection    *mongo.Collection
}

type uniqueIndexInfo struct {
	coll *mongo.Collection
	kv   map[string]int32
}

func NewMongoService(uri string) (*MongoService, error) {
	// create the base context, the context in which your application runs
	//https://www.mongodb.com/blog/post/quick-start-golang-mongodb-starting-and-setup
	//10seconds = timeout duration that we want to use when trying to connect.
	//ctx, cancel := context.WithCancel(context.Background())
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	client, err := mongo.NewClient(options.Client().ApplyURI(uri))
	if err != nil {
		return nil, err
	}

	// add disconnect call here
	//todo can't call defer here because once method returns db connection closes. When to close the connection?
	//https://codereview.stackexchange.com/questions/241739/connection-to-mongodb-in-golang
	//defer func() {
	//	zap.S().Infof("closing database connection ...")
	//	if err = client.Disconnect(ctx); err != nil {
	//		panic(err)
	//	}
	//}()

	if err := client.Connect(ctx); err != nil {
		return nil, err
	}

	// Ping the primary to check connection establishment
	if err := client.Ping(ctx, readpref.Primary()); err != nil {
		return nil, err
	}
	zap.S().Infof("Successfully connected to database and pinged")

	db := client.Database(DatabaseName)
	mongoDB := &MongoService{
		client:            client,
		database:          db,
		datasetCollection: db.Collection(DatasetCollection),
		designCollection:  db.Collection(DesignCollection),
		jobCollection:     db.Collection(JobCollection),
		taskCollection:    db.Collection(TaskCollection),
	}

	uiiList := []uniqueIndexInfo{
		{
			mongoDB.datasetCollection,
			map[string]int32{"userid": orderAscend, "url": orderAscend},
		},
		{
			mongoDB.designCollection,
			map[string]int32{"userid": orderAscend, "id": orderAscend},
		},
		{
			mongoDB.taskCollection,
			map[string]int32{"jobid": orderAscend, "agentid": orderAscend},
		},
	}
	for _, uii := range uiiList {
		if err := createUniqueIndex(uii.coll, ctx, uii.kv); err != nil {
			return nil, err
		}
	}

	return mongoDB, nil
}

func createUniqueIndex(coll *mongo.Collection, ctx context.Context, kv map[string]int32) error {
	keysDoc := bsonx.Doc{}

	for key, val := range kv {
		if val != orderAscend && val != orderDescend {
			val = orderAscend
		}
		keysDoc = keysDoc.Append(key, bsonx.Int32(val))
	}

	model := mongo.IndexModel{
		Keys: keysDoc,
		// create UniqueIndex option
		Options: options.Index().SetUnique(true),
	}

	index, err := coll.Indexes().CreateOne(ctx, model)
	if err != nil {
		zap.S().Debugf("Failed to created index for %s: %v", coll.Name(), err)
		return err
	}

	zap.S().Infof("Created index %s for %s successfully", index, coll.Name())

	return nil
}
