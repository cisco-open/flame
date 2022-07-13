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
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.uber.org/zap"

	"github.com/cisco-open/flame/pkg/openapi"
)

// RegisterCompute creates a new cluster compute specification and returns ComputeStatus
func (db *MongoService) RegisterCompute(computeSpec openapi.ComputeSpec) (openapi.ComputeStatus, error) {
	// First check if the compute was previously registered
	filter := bson.M{"computeid": computeSpec.ComputeId}
	checkResult := db.computeCollection.FindOne(context.TODO(), filter)
	if checkResult.Err() == mongo.ErrNoDocuments {
		// If it was not registered previously, need to register
		result, err := db.computeCollection.InsertOne(context.TODO(), computeSpec)
		if err != nil {
			zap.S().Errorf("Failed to register new compute in database: result: %v, error: %v", result, err)

			return openapi.ComputeStatus{}, ErrorCheck(err)
		}

		computeStatus := openapi.ComputeStatus{
			ComputeId:    computeSpec.ComputeId,
			State:        openapi.UP,
			RegisteredAt: time.Now(),
		}

		updateTime, err := db.UpdateComputeStatus(computeStatus.ComputeId, computeStatus)
		if err != nil {
			zap.S().Errorf("Failed to update compute status: %v", err)

			return openapi.ComputeStatus{}, ErrorCheck(err)
		}

		zap.S().Infof("Successfully registered a new compute cluster with computeID %s", computeStatus.ComputeId)
		computeStatus.UpdatedAt = updateTime

		return computeStatus, err
	} else {
		var currentDocument openapi.ComputeSpec
		err := checkResult.Decode(&currentDocument)
		if err != nil {
			zap.S().Errorf("Failed to parse currentDocument: %v", err)

			return openapi.ComputeStatus{}, ErrorCheck(err)
		}
		if currentDocument.ApiKey != computeSpec.ApiKey {
			errMsg := fmt.Sprintf("ComputeId %v already in use. Use a different computeId and try again", computeSpec.ComputeId)
			zap.S().Errorf(errMsg)

			return openapi.ComputeStatus{}, fmt.Errorf(errMsg)
		}
		return openapi.ComputeStatus{}, nil
	}
}

// UpdateComputeStatus update compute cluster's status
func (db *MongoService) UpdateComputeStatus(computeId string, computeStatus openapi.ComputeStatus) (time.Time, error) {
	dateKey := ""
	updateTime := time.Time{}
	switch computeStatus.State {
	case openapi.UP:
		fallthrough
	case openapi.DOWN:
		fallthrough
	case openapi.MAINTENANCE:
		dateKey = updatedAt

	default:
		return time.Time{}, fmt.Errorf("unknown state: %s", computeStatus.State)
	}

	setElements := bson.M{"state": computeStatus.State,
		"registeredat": computeStatus.RegisteredAt}
	if dateKey != "" {
		updateTime = time.Now()
		setElements[dateKey] = updateTime
	}

	filter := bson.M{"computeid": computeId}
	update := bson.M{"$set": setElements}

	updatedDoc := openapi.ComputeStatus{}
	err := db.computeCollection.FindOneAndUpdate(context.TODO(), filter, update).Decode(&updatedDoc)
	if err != nil {
		return time.Time{}, err
	}

	return updateTime, nil
}
