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

// CreateDesignSchema adds schema design to the design template information
func (db *MongoService) CreateDesignSchema(userId string, designId string, ds openapi.DesignSchema) error {
	zap.S().Debugf("Creating design schema request for userId: %s | designId: %s", userId, designId)

	db.incrementSchemaRevision(userId, designId, &ds)

	var updatedDoc openapi.Design
	filter := bson.M{util.DBFieldUserId: userId, util.DBFieldId: designId}
	update := bson.M{"$set": bson.M{util.DBFieldSchema: ds}}

	err := db.designCollection.FindOneAndUpdate(context.TODO(), filter, update).Decode(&updatedDoc)
	if err != nil {
		zap.S().Errorf("Failed to insert the design template. %v", err)
		return ErrorCheck(err)
	}

	return nil
}

func (db *MongoService) GetDesignSchema(userId string, designId string) (openapi.DesignSchema, error) {
	zap.S().Debugf("Getting schema details for userId: %s | designId: %s", userId, designId)

	filter := bson.M{util.DBFieldUserId: userId, util.DBFieldId: designId}

	updatedDoc := openapi.Design{}
	err := db.designCollection.FindOne(context.TODO(), filter).Decode(&updatedDoc)
	if err != nil {
		err = ErrorCheck(err)
		zap.S().Errorf("Failed to fetch design schema information: %v", err)
		return openapi.DesignSchema{}, err
	}

	if updatedDoc.Schema.Revision == 0 {
		return openapi.DesignSchema{}, fmt.Errorf("no design schema found")
	}

	return updatedDoc.Schema, nil
}

func (db *MongoService) UpdateDesignSchema(userId string, designId string, ds openapi.DesignSchema) error {
	zap.S().Debugf("Updating design schema request for userId: %s | designId: %s", userId, designId)

	db.incrementSchemaRevision(userId, designId, &ds)

	filter := bson.M{util.DBFieldUserId: userId, util.DBFieldId: designId}
	update := bson.M{"$set": bson.M{util.DBFieldSchema: ds}}

	var updatedDoc openapi.Design
	err := db.designCollection.FindOneAndUpdate(context.TODO(), filter, update).Decode(&updatedDoc)
	if err != nil {
		zap.S().Errorf("Failed to update the design template: %v", err)
		return ErrorCheck(err)
	}

	return nil
}

// DeleteDesignSchema deletes design schema for the given design Id
func (db *MongoService) DeleteDesignSchema(userId string, designId string) error {
	zap.S().Debugf("delete schema: user: %v, design ID: %v", userId, designId)

	emptySchema := openapi.DesignSchema{}
	filter := bson.M{util.DBFieldUserId: userId, util.DBFieldId: designId}
	update := bson.M{"$set": bson.M{util.DBFieldSchema: emptySchema}}

	var updatedDoc openapi.Design
	err := db.designCollection.FindOneAndUpdate(context.TODO(), filter, update).Decode(&updatedDoc)
	if err != nil {
		return fmt.Errorf("failed to delete design schema: %v", err)
	}

	zap.S().Debugf("Deleted schema of design: %s", designId)

	return nil
}

func (db *MongoService) incrementSchemaRevision(userId string, designId string, ds *openapi.DesignSchema) {
	schema, err := db.GetDesignSchema(userId, designId)
	if err != nil {
		ds.Revision = 1
	} else {
		ds.Revision = schema.Revision + 1
	}
}
