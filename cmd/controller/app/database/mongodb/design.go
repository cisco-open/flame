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
	"go.uber.org/zap"

	"github.com/cisco-open/flame/pkg/openapi"
	"github.com/cisco-open/flame/pkg/util"
)

// CreateDesign create a new design entry in the database
func (db *MongoService) CreateDesign(userId string, design openapi.Design) error {
	_, err := db.designCollection.InsertOne(context.TODO(), design)
	if err != nil {
		err = ErrorCheck(err)
		zap.S().Errorf("Failed to create new design: %v", err)

		return err
	}

	zap.S().Debugf("New design for user: %s inserted with ID: %s", userId, design.Id)

	return nil
}

func (db *MongoService) DeleteDesign(userId string, designId string, forceDelete bool) error {
	filter := bson.M{util.DBFieldUserId: userId, util.DBFieldDesignId: designId}

	count, err := db.jobCollection.CountDocuments(context.TODO(), filter)

	if err != nil {
		zap.S().Errorf("Error counting jobs: %v", err)
	}

	if count == 0 {
		zap.S().Debugf("No jobs found. Design can be safely deleted.")
	} else if !forceDelete {
		return fmt.Errorf("design used in job: design id: %s | jobs count %d", designId, count)
	}

	zap.S().Debugf("Delete design : %v, %v", userId, designId)

	updateRes, err := db.designCollection.DeleteOne(context.TODO(),
		bson.M{util.DBFieldUserId: userId, util.DBFieldId: designId})

	if err != nil {
		return fmt.Errorf("failed to delete design error: %v", err)
	}

	if updateRes.DeletedCount == 0 {
		return fmt.Errorf("design id %s not found", designId)
	}

	zap.S().Debugf("Successfully deleted design: %v", updateRes)

	return nil
}

// GetDesign returns the details about the given design id
func (db *MongoService) GetDesign(userId string, designId string) (openapi.Design, error) {
	zap.S().Debugf("Get design information for user: %s | desginId: %s", userId, designId)

	var design openapi.Design

	filter := bson.M{util.DBFieldUserId: userId, util.DBFieldId: designId}
	err := db.designCollection.FindOne(context.TODO(), filter).Decode(&design)
	if err != nil {
		err = ErrorCheck(err)
		zap.S().Errorf("Failed to fetch design template for %s: %v", designId, err)

		return design, err
	}

	return design, nil
}

// GetDesigns get a lists of all the designs created by the user
// TODO: update the method to implement a limit next based cursor
func (db *MongoService) GetDesigns(userId string, limit int32) ([]openapi.DesignInfo, error) {
	zap.S().Debugf("Get list of designs for user: %s | limit: %d", userId, limit)

	filter := bson.M{util.DBFieldUserId: userId}
	cursor, err := db.designCollection.Find(context.TODO(), filter)
	if err != nil {
		err = ErrorCheck(err)
		zap.S().Errorf("Error while getting list of designs. %v", err)
		return nil, err
	}

	defer cursor.Close(context.TODO())
	var designInfoList []openapi.DesignInfo

	for cursor.Next(context.TODO()) {
		var designInfo openapi.DesignInfo
		if err = cursor.Decode(&designInfo); err != nil {
			err = ErrorCheck(err)
			zap.S().Errorf("Failed to decode design info: %v", err)

			return nil, err
		}

		designInfoList = append(designInfoList, designInfo)
	}

	return designInfoList, nil
}

func (db *MongoService) UpdateDesign(userId string, designId string,
	design openapi.DesignInfo) (openapi.DesignInfo, error) {
	zap.S().Debugf("Updating dataset request for userId: %s | datasetId: %s", userId, designId)

	filter := bson.M{util.DBFieldUserId: userId, util.DBFieldId: designId}
	update := bson.M{"$set": design}

	var updatedDoc openapi.DesignInfo

	err := db.designCollection.FindOneAndUpdate(context.TODO(), filter, update).Decode(&updatedDoc)

	if err != nil {
		zap.S().Errorf("Failed to update the design: %v", err)
		return design, ErrorCheck(err)
	}

	return design, nil
}
