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

package mongodb

import (
	"context"
	"fmt"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.uber.org/zap"

	"github.com/cisco-open/flame/pkg/openapi"
	"github.com/cisco-open/flame/pkg/util"
)

// CreateDataset creates a new dataset entry in the database
func (db *MongoService) CreateDataset(userId string, dataset openapi.DatasetInfo) (string, error) {
	// override user in the DatasetInfo struct to prevent wrong user name is recorded in the db
	dataset.UserId = userId

	result, err := db.datasetCollection.InsertOne(context.TODO(), dataset)
	if err != nil {
		err = ErrorCheck(err)
		zap.S().Errorf("Failed to create new dataset: %v", err)

		return "", err
	}

	err = db.setDatasetId(result.InsertedID.(primitive.ObjectID))
	if err != nil {
		zap.S().Errorf("Failed to set Id: %v", err)
		return "", err
	}

	dataset.Id = GetStringID(result.InsertedID.(primitive.ObjectID))

	zap.S().Debugf("New dataset for user: %s inserted with ID: %s", userId, dataset.Id)

	return dataset.Id, nil
}

func (db *MongoService) setDatasetId(docId primitive.ObjectID) error {
	filter := bson.M{util.DBFieldMongoID: docId}
	update := bson.M{"$set": bson.M{util.DBFieldId: GetStringID(docId)}}

	updatedDoc := openapi.DatasetInfo{}
	err := db.datasetCollection.FindOneAndUpdate(context.TODO(), filter, update).Decode(&updatedDoc)
	if err != nil {
		return ErrorCheck(err)
	}

	return nil
}

func (db *MongoService) GetDatasetById(datasetId string) (openapi.DatasetInfo, error) {
	zap.S().Infof("get dataset info for datasetId: %s", datasetId)

	filter := bson.M{util.DBFieldMongoID: ConvertToObjectID(datasetId)}
	var datasetInfo openapi.DatasetInfo
	err := db.datasetCollection.FindOne(context.TODO(), filter).Decode(&datasetInfo)
	if err != nil {
		zap.S().Warnf("failed to fetch dataset info: %v", err)

		return openapi.DatasetInfo{}, fmt.Errorf("dataset %s - %v", datasetId, ErrorCheck(err))
	}

	return datasetInfo, nil
}

func (db *MongoService) GetDatasets(userId string, limit int32) ([]openapi.DatasetInfo, error) {
	zap.S().Infof("Get datasets info for user: %s", userId)

	filter := bson.M{util.DBFieldIsPublic: true}
	if userId != "" {
		filter = bson.M{util.DBFieldUserId: userId}
	}

	cursor, err := db.datasetCollection.Find(context.TODO(), filter)
	if err != nil {
		zap.S().Warnf("Failed to fetch datasets info: %v", err)

		return nil, ErrorCheck(err)
	}

	defer cursor.Close(context.TODO())
	var datasetInfoList []openapi.DatasetInfo

	for cursor.Next(context.TODO()) {
		var datasetInfo openapi.DatasetInfo
		if err = cursor.Decode(&datasetInfo); err != nil {
			err = ErrorCheck(err)
			zap.S().Errorf("Failed to decode dataset info: %v", err)

			return nil, err
		}

		datasetInfoList = append(datasetInfoList, datasetInfo)
	}

	return datasetInfoList, nil
}
