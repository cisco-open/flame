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
	"go.mongodb.org/mongo-driver/mongo/options"
	"go.uber.org/zap"

	"github.com/cisco-open/flame/pkg/openapi"
	"github.com/cisco-open/flame/pkg/util"
)

// CreateDesignSchema adds schema design to the design template information
func (db *MongoService) CreateDesignSchema(userId string, designId string, ds openapi.DesignSchema) error {
	zap.S().Debugf("Creating design schema request for userId: %s | designId: %s", userId, designId)

	latestSchema, err := db.GetDesignSchema(userId, designId, latestVersion)
	if err != nil {
		ds.Version = "1"
	} else {
		ds.Version = IncrementVersion(latestSchema.Version)
	}

	var updatedDoc openapi.Design
	filter := bson.M{util.DBFieldUserId: userId, util.DBFieldId: designId}
	update := bson.M{"$push": bson.M{"schemas": ds}}

	err = db.designCollection.FindOneAndUpdate(context.TODO(), filter, update).Decode(&updatedDoc)
	if err != nil {
		zap.S().Errorf("Failed to insert the design template. %v", err)
		return ErrorCheck(err)
	}

	return nil
}

func (db *MongoService) GetDesignSchema(userId string, designId string, version string) (openapi.DesignSchema, error) {
	zap.S().Debugf("Getting schema details for userId: %s | designId: %s | schema version: %s",
		userId, designId, version)

	filter := bson.M{util.DBFieldUserId: userId, util.DBFieldId: designId, "schemas.version": version}
	opts := options.FindOne().SetProjection(bson.M{"schemas.$": 1})

	if version == latestVersion {
		filter = bson.M{util.DBFieldUserId: userId, util.DBFieldId: designId}
		projection := bson.M{"schemas": bson.M{"$slice": -1}}
		opts = options.FindOne().SetProjection(projection)
	}

	updatedDoc := openapi.Design{}
	err := db.designCollection.FindOne(context.TODO(), filter, opts).Decode(&updatedDoc)
	if err != nil {
		err = ErrorCheck(err)
		zap.S().Errorf("Failed to fetch design schema information: %v", err)
		return openapi.DesignSchema{}, err
	}

	if len(updatedDoc.Schemas) == 0 {
		return openapi.DesignSchema{}, fmt.Errorf("no design schema found")
	}

	return updatedDoc.Schemas[0], err
}

func (db *MongoService) GetDesignSchemas(userId string, designId string) ([]openapi.DesignSchema, error) {
	zap.S().Debugf("Getting schema details for userId: %s | designId: %s", userId, designId)

	filter := bson.M{util.DBFieldUserId: userId, util.DBFieldId: designId}
	opts := options.FindOne().SetProjection(bson.M{"schemas": 1})

	updatedDoc := openapi.Design{}
	err := db.designCollection.FindOne(context.TODO(), filter, opts).Decode(&updatedDoc)
	if err != nil {
		err = ErrorCheck(err)
		zap.S().Errorf("Failed to fetch design schema information: %v", err)
		return []openapi.DesignSchema{}, err
	}

	return updatedDoc.Schemas, err
}

func (db *MongoService) UpdateDesignSchema(userId string, designId string, version string, ds openapi.DesignSchema) error {
	zap.S().Debugf("Updating design schema request for userId: %s | designId: %s | version: %s", userId, designId, version)

	// set version in the design schema
	ds.Version = version

	filter := bson.M{util.DBFieldUserId: userId, util.DBFieldId: designId, "schemas.version": version}
	update := bson.M{"$set": bson.M{"schemas.$": ds}}

	var updatedDoc openapi.Design
	err := db.designCollection.FindOneAndUpdate(context.TODO(), filter, update).Decode(&updatedDoc)
	if err != nil {
		zap.S().Errorf("Failed to update the design template: %v", err)
		return ErrorCheck(err)
	}

	return nil
}

// DeleteDesignSchema delete design schema for the given version and design Id
func (db *MongoService) DeleteDesignSchema(userId string, designId string, version string) error {
	zap.S().Debugf("delete design schema : %v, %v,%v", userId, designId, version)

	updateRes, err := db.designCollection.UpdateOne(context.TODO(),
		bson.M{util.DBFieldUserId: userId, util.DBFieldId: designId},
		bson.M{"$pull": bson.M{util.DBFieldSchemas: bson.M{"version": version}}})
	if err != nil {
		return fmt.Errorf("failed to delete design schema deleted error: %v", err)
	}
	if updateRes.ModifiedCount == 0 {
		return fmt.Errorf("failed to delete design schema, schema version %s not found. deleted schema count: %#v",
			version, updateRes)
	}
	zap.S().Debugf("successfully deleted design schema: %#v", updateRes)
	return nil
}
