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

	"go.mongodb.org/mongo-driver/bson"
	"go.uber.org/zap"

	"github.com/cisco/fledge/pkg/openapi"
	"github.com/cisco/fledge/pkg/util"
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

// GetDesign returns the details about the given design id
func (db *MongoService) GetDesign(userId string, designId string) (openapi.Design, error) {
	zap.S().Debugf("Get design information for user: %s | desginId: %s", userId, designId)

	var design openapi.Design

	filter := bson.M{util.DBFieldUserId: userId, util.DBFieldId: designId}
	err := db.designCollection.FindOne(context.TODO(), filter).Decode(&design)
	if err != nil {
		err = ErrorCheck(err)
		zap.S().Errorf("Failed to fetch design template information: %v", err)
	}

	return design, err
}

// GetDesigns get a lists of all the designs created by the user
// TODO: update the method to implement a limit next based cursor
func (db *MongoService) GetDesigns(userId string, limit int32) ([]openapi.DesignInfo, error) {
	zap.S().Debugf("Get list of designs for user: %s | limit: %d", userId, limit)

	filter := bson.M{util.DBFieldUserId: userId}
	cursor, err := db.designCollection.Find(context.TODO(), filter)
	if err != nil {
		err = ErrorCheck(err)
		zap.S().Errorf("error while getting list of designs. %v", err)
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
