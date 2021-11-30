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
	"fmt"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.uber.org/zap"

	"github.com/cisco/fledge/pkg/openapi"
	"github.com/cisco/fledge/pkg/util"
)

// CreateJob creates a new job specification and returns JobStatus
func (db *MongoService) CreateJob(userId string, jobSpec openapi.JobSpec) (openapi.JobStatus, error) {
	// override userId in jobSpec to prevent an incorrect record in the db
	jobSpec.UserId = userId

	result, err := db.jobCollection.InsertOne(context.TODO(), jobSpec)
	if err != nil {
		zap.S().Warnf("Failed to create a new job: %v", err)

		return openapi.JobStatus{}, ErrorCheck(err)
	}

	jobStatus := openapi.JobStatus{
		Id:    GetStringID(result.InsertedID),
		State: openapi.READY,
	}

	err = db.UpdateJobStatus(userId, jobStatus.Id, jobStatus)
	if err != nil {
		zap.S().Warnf("Failed to update job status: %v", err)

		return openapi.JobStatus{}, ErrorCheck(err)
	}

	zap.S().Infof("Successfully created a new job for user %s with job ID %s", userId, jobStatus.Id)

	return jobStatus, err
}

func (db *MongoService) DeleteJob(userId string, jobId string) error {
	return nil
}

func (db *MongoService) GetJob(userId string, jobId string) (openapi.JobSpec, error) {
	zap.S().Infof("get job specification for userId: %s with jobId: %s", userId, jobId)

	filter := bson.M{util.DBFieldMongoID: ConvertToObjectID(jobId), util.DBFieldUserId: userId}
	var jobSpec openapi.JobSpec
	err := db.jobCollection.FindOne(context.TODO(), filter).Decode(&jobSpec)
	if err != nil {
		zap.S().Warnf("failed to fetch job specification: %v", err)

		return openapi.JobSpec{}, ErrorCheck(err)
	}

	return jobSpec, nil
}

func (db *MongoService) GetJobById(jobId string) (openapi.JobSpec, error) {
	zap.S().Infof("get job specification for jobId: %s", jobId)

	filter := bson.M{util.DBFieldMongoID: ConvertToObjectID(jobId)}
	var jobSpec openapi.JobSpec
	err := db.jobCollection.FindOne(context.TODO(), filter).Decode(&jobSpec)
	if err != nil {
		zap.S().Warnf("failed to fetch job specification: %v", err)

		return openapi.JobSpec{}, ErrorCheck(err)
	}

	return jobSpec, nil
}

func (db *MongoService) GetJobStatus(userId string, jobId string) (openapi.JobStatus, error) {
	zap.S().Debugf("get job status for userId: %s with jobId: %s", userId, jobId)

	filter := bson.M{util.DBFieldMongoID: ConvertToObjectID(jobId), util.DBFieldUserId: userId}
	jobStatus := openapi.JobStatus{}
	err := db.jobCollection.FindOne(context.TODO(), filter).Decode(&jobStatus)
	if err != nil {
		zap.S().Warnf("failed to fetch job status: %v", err)

		return openapi.JobStatus{}, ErrorCheck(err)
	}

	return jobStatus, nil
}

func (db *MongoService) GetJobs(userId string, limit int32) ([]openapi.JobStatus, error) {
	zap.S().Infof("get status of all jobs owned by user: %s", userId)

	filter := bson.M{util.DBFieldUserId: userId}

	cursor, err := db.jobCollection.Find(context.TODO(), filter)
	if err != nil {
		zap.S().Warnf("failed to fetch jobs' status: %v", err)

		return nil, ErrorCheck(err)
	}

	defer cursor.Close(context.TODO())
	var jobStatusList []openapi.JobStatus

	for cursor.Next(context.TODO()) {
		var jobStatus openapi.JobStatus
		if err = cursor.Decode(&jobStatus); err != nil {
			zap.S().Errorf("Failed to decode job status: %v", err)

			return nil, ErrorCheck(err)
		}

		jobStatusList = append(jobStatusList, jobStatus)
	}

	return jobStatusList, nil
}

func (db *MongoService) UpdateJob(userId string, jobId string, jobSpec openapi.JobSpec) error {
	jobStatus, err := db.GetJobStatus(userId, jobId)
	if err != nil {
		return err
	}

	switch jobStatus.State {
	case openapi.READY:
		fallthrough
	case openapi.FAILED:
		fallthrough
	case openapi.TERMINATED:
		fallthrough
	case openapi.COMPLETED:
		// Do nothing

	case openapi.STARTING:
		fallthrough
	case openapi.APPLYING:
		fallthrough
	case openapi.RUNNING:
		fallthrough
	case openapi.DEPLOYING:
		fallthrough
	case openapi.STOPPING:
		fallthrough
	default:
		err = fmt.Errorf("The current job state (%s) is not an updatable state", jobStatus.State)
		return err
	}

	jobSpec.Id = jobId
	jobSpec.UserId = userId

	filter := bson.M{util.DBFieldMongoID: ConvertToObjectID(jobId), util.DBFieldUserId: userId}
	update := bson.M{"$set": jobSpec}

	updatedDoc := openapi.JobSpec{}
	err = db.jobCollection.FindOneAndUpdate(context.TODO(), filter, update).Decode(&updatedDoc)
	if err != nil {
		return ErrorCheck(err)
	}

	return nil
}

// UpdateJobStatus update Job's status
func (db *MongoService) UpdateJobStatus(userId string, jobId string, jobStatus openapi.JobStatus) error {
	dateKey := ""
	switch jobStatus.State {
	case openapi.READY:
		dateKey = "createdat"

	case openapi.STARTING:
		dateKey = "startedat"

	case openapi.APPLYING:
		dateKey = "updatedat"

	case openapi.FAILED:
		fallthrough
	case openapi.TERMINATED:
		fallthrough
	case openapi.COMPLETED:
		dateKey = "endedat"

	case openapi.RUNNING:
		fallthrough
	case openapi.DEPLOYING:
		fallthrough
	case openapi.STOPPING:
		dateKey = ""

	default:
		return fmt.Errorf("unknown state: %s", jobStatus.State)
	}

	setElements := bson.M{util.DBFieldId: jobId, "state": jobStatus.State}
	if dateKey != "" {
		setElements[dateKey] = time.Now()
	}

	filter := bson.M{util.DBFieldMongoID: ConvertToObjectID(jobId)}
	update := bson.M{"$set": setElements}

	updatedDoc := openapi.JobStatus{}
	err := db.jobCollection.FindOneAndUpdate(context.TODO(), filter, update).Decode(&updatedDoc)
	if err != nil {
		return ErrorCheck(err)
	}

	return nil
}
