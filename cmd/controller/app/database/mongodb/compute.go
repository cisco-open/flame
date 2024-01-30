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
	"errors"
	"fmt"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.uber.org/zap"

	"github.com/cisco-open/flame/pkg/openapi"
	"github.com/cisco-open/flame/pkg/util"
)

// RegisterCompute creates a new cluster compute specification and returns ComputeStatus
func (db *MongoService) RegisterCompute(computeSpec openapi.ComputeSpec) (openapi.ComputeStatus, error) {
	// First check if the compute was previously registered
	filter := bson.M{util.DBFieldComputeId: computeSpec.ComputeId}
	checkResult := db.computeCollection.FindOne(context.TODO(), filter)
	if (checkResult.Err() != nil) && (checkResult.Err() != mongo.ErrNoDocuments) {
		errMsg := fmt.Sprintf("Failed to register compute : %v", checkResult.Err())
		zap.S().Errorf(errMsg)

		return openapi.ComputeStatus{}, ErrorCheck(checkResult.Err())
	}
	if checkResult.Err() == mongo.ErrNoDocuments {
		// If it was not registered previously, need to register
		result, err := db.computeCollection.InsertOne(context.TODO(), computeSpec)
		if err != nil {
			errMsg := fmt.Sprintf("Failed to register new compute in database: result: %v, error: %v", result, err)
			zap.S().Errorf(errMsg)
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

func (db *MongoService) GetAllComputes(adminId string) ([]openapi.ComputeSpec, error) {
	zap.S().Infof("Get computes info for admin: %s", adminId)

	if adminId == "" {
		return nil, errors.New("Admin ID parameter is required")
	}
	filter := bson.M{util.DBFieldIsPublic: true, util.DBFieldAdminId: adminId}

	cursor, err := db.computeCollection.Find(context.TODO(), filter)
	if err != nil {
		zap.S().Warnf("Failed to fetch computes info: %v", err)

		return nil, ErrorCheck(err)
	}

	defer cursor.Close(context.TODO())
	var computeInfoList []openapi.ComputeSpec

	for cursor.Next(context.TODO()) {
		var computesInfo openapi.ComputeSpec
		if err = cursor.Decode(&computesInfo); err != nil {
			err = ErrorCheck(err)
			zap.S().Errorf("Failed to decode compute info: %v", err)

			return nil, err
		}

		computeInfoList = append(computeInfoList, computesInfo)
	}

	zap.S().Debugf("compute list db: %s ", computeInfoList)

	return computeInfoList, nil
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

	filter := bson.M{util.DBFieldComputeId: computeId}
	update := bson.M{"$set": setElements}

	updatedDoc := openapi.ComputeStatus{}
	err := db.computeCollection.FindOneAndUpdate(context.TODO(), filter, update).Decode(&updatedDoc)
	if err != nil {
		return time.Time{}, err
	}

	return updateTime, nil
}

func (db *MongoService) GetComputeIdsByRegion(region string) ([]string, error) {
	zap.S().Infof("get all computes in the region: %s", region)

	filter := bson.M{util.DBFieldComputeRegion: region}
	cursor, err := db.computeCollection.Find(context.TODO(), filter)
	if err != nil {
		errMsg := fmt.Sprintf("failed to fetch computes in the region: %s, err : %v", region, err)
		zap.S().Errorf(errMsg)

		return []string{}, fmt.Errorf(errMsg)
	}

	defer cursor.Close(context.TODO())
	var computeIdList []string

	for cursor.Next(context.TODO()) {
		var computeSpec openapi.ComputeSpec
		if err = cursor.Decode(&computeSpec); err != nil {
			errMsg := fmt.Sprintf("failed to decode compute spec with error: %v", err)
			zap.S().Errorf(errMsg)

			return []string{}, ErrorCheck(err)
		}

		computeIdList = append(computeIdList, computeSpec.ComputeId)
	}

	if len(computeIdList) == 0 {
		errMsg := fmt.Sprintf("could not find any computes for the region: %s", region)
		zap.S().Errorf(errMsg)

		return []string{}, fmt.Errorf(errMsg)
	}
	return computeIdList, nil
}

func (db *MongoService) GetComputeById(computeId string) (openapi.ComputeSpec, error) {
	filter := bson.M{util.DBFieldComputeId: computeId}
	checkResult := db.computeCollection.FindOne(context.TODO(), filter)
	if checkResult.Err() != nil {
		errMsg := fmt.Sprintf("failed to find a compute with computeId: %s", computeId)
		zap.S().Errorf(errMsg)
		return openapi.ComputeSpec{}, fmt.Errorf(errMsg)
	}

	var currentDocument openapi.ComputeSpec
	err := checkResult.Decode(&currentDocument)
	if err != nil {
		errMsg := fmt.Sprintf("Failed to parse currentDocument: %v", err)
		zap.S().Errorf(errMsg)
		return openapi.ComputeSpec{}, fmt.Errorf(errMsg)
	}
	return currentDocument, nil
}

func (db *MongoService) UpdateDeploymentStatus(computeId string, jobId string, agentStatuses map[string]openapi.AgentState) error {
	filter := bson.M{util.DBFieldJobId: jobId, util.DBFieldComputeId: computeId}
	result := db.deploymentCollection.FindOne(context.TODO(), filter)
	// update fields if document is already present
	if result.Err() == nil {
		result = db.deploymentCollection.FindOneAndUpdate(context.TODO(),
			filter,
			bson.M{"$set": bson.M{util.DBFieldAgentStatuses: agentStatuses}})
		if err := result.Err(); err != nil {
			err = ErrorCheck(err)
			zap.S().Errorf("Failed to update new deployment status: %v", err)
			return err
		}
	} else {
		_, err := db.deploymentCollection.InsertOne(context.TODO(),
			bson.M{util.DBFieldJobId: jobId,
				util.DBFieldComputeId:     computeId,
				util.DBFieldAgentStatuses: agentStatuses,
			})
		if err != nil {
			err = ErrorCheck(err)
			zap.S().Errorf("Failed to create new deployment status: %v", err)
			return err
		}
	}
	zap.S().Debugf("New deployment status for jobid: %s inserted", jobId)
	return nil
}
