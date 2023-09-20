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
	"go.uber.org/zap"

	"github.com/cisco-open/flame/pkg/openapi"
	"github.com/cisco-open/flame/pkg/util"
)

const (
	hiddenMessage = "hidden by system"
)

// CreateJob creates a new job specification and returns JobStatus
func (db *MongoService) CreateJob(userId string, jobSpec openapi.JobSpec) (openapi.JobStatus, error) {
	schema, err := db.GetDesignSchema(userId, jobSpec.DesignId)
	if err != nil {
		zap.S().Warnf("Failed to obtain design schema: %v", err)

		return openapi.JobStatus{}, ErrorCheck(err)
	}

	codeApiRes, err := db.GetDesignCodeRevision(userId, jobSpec.DesignId)
	if err != nil {
		zap.S().Warnf("Failed to obtain code revision: %v", err)

		return openapi.JobStatus{}, ErrorCheck(err)
	}

	// get schema's and code's revision numbers
	// so that these numbers can be tracked
	// for user's debugging purposes
	jobSpec.SchemaRevision = schema.Revision
	jobSpec.CodeRevision = codeApiRes.Revision
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
	zap.S().Infof("Deleting job: %s", jobId)
	filter := bson.M{util.DBFieldUserId: userId, util.DBFieldId: jobId}

	_, err := db.jobCollection.DeleteOne(context.TODO(), filter)
	if err != nil {
		errMsg := fmt.Sprintf("failed to delete job: %v", err)
		zap.S().Warn(errMsg)
		return fmt.Errorf("%s", errMsg)
	}

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

func (db *MongoService) GetJobsByCompute(computeId string) ([]openapi.JobStatus, error) {
	zap.S().Infof("get status of all jobs by compute that have finished")

	ctx := context.Background()
	pipeline := []bson.M{
		{
			"$match": bson.M{
				"state": bson.M{
					"$eq": "running",
				},
			},
		},
		{
			"$lookup": bson.M{
				"from": "t_task",
				"let":  bson.M{"id": "$id"},
				"pipeline": []bson.M{
					{
						"$match": bson.M{"$expr": bson.M{"$eq": []string{"$jobid", "$$id"}}},
					},
					{
						"$match": bson.M{"$expr": bson.M{"$eq": []string{"$jobid", computeId}}},
					},
					{
						"$project": bson.M{"computeid": 1},
					},
				},
				"as": "tasks",
			},
		},
		{
			"$match": bson.M{
				"tasks.1": bson.M{
					"$exists": true,
				},
			},
		},
	}

	cursor, err := db.jobCollection.Aggregate(ctx, pipeline)
	if err != nil {
		zap.S().Warnf("failed to fetch jobs' status: %v", err)

		return nil, ErrorCheck(err)
	}

	defer cursor.Close(ctx)
	var jobStatusList []openapi.JobStatus

	for cursor.Next(ctx) {
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
		err = fmt.Errorf("the current job state (%s) is not an updatable state", jobStatus.State)
		return err
	}

	schema, err := db.GetDesignSchema(userId, jobSpec.DesignId)
	if err != nil {
		zap.S().Warnf("Failed to obtain design schema: %v", err)

		return ErrorCheck(err)
	}

	codeApiRes, err := db.GetDesignCodeRevision(userId, jobSpec.DesignId)
	if err != nil {
		zap.S().Warnf("Failed to obtain code revision: %v", err)

		return ErrorCheck(err)
	}

	// get schema's and code's revision numbers
	// so that these numbers can be tracked
	// for user's debugging purposes
	jobSpec.SchemaRevision = schema.Revision
	jobSpec.CodeRevision = codeApiRes.Revision
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
		dateKey = updatedAt

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

func (db *MongoService) jobExists(userId string, jobId string) error {
	filter := bson.M{util.DBFieldId: jobId, util.DBFieldUserId: userId}

	return db.jobCollection.FindOne(context.TODO(), filter).Err()
}

func (db *MongoService) GetTaskInfo(userId string, jobId string, taskId string) (openapi.TaskInfo, error) {
	zap.S().Infof("Get info of task %s in job %s owned by user: %s", taskId, jobId, userId)

	if err := db.jobExists(userId, jobId); err != nil {
		zap.S().Warnf("failed to find a matching job", err)
		return openapi.TaskInfo{}, ErrorCheck(err)
	}

	// found a job; so we can proceed to fetch the task info

	filter := bson.M{util.DBFieldJobId: jobId, util.DBFieldTaskId: taskId}

	var taskInfo openapi.TaskInfo
	err := db.taskCollection.FindOne(context.TODO(), filter).Decode(&taskInfo)
	if err != nil {
		zap.S().Warnf("failed to fetch task info: %v", err)

		return openapi.TaskInfo{}, ErrorCheck(err)
	}

	// strip off the key field in the data structure
	// This is to protect key information
	taskInfo.Key = hiddenMessage

	return taskInfo, nil
}

func (db *MongoService) GetTasksInfo(userId string, jobId string, limit int32, inclKey bool) ([]openapi.TaskInfo, error) {
	tasksInfoList, err := db.GetTasksInfoGeneric(userId, jobId, limit, inclKey, false)
	if err != nil {
		return []openapi.TaskInfo{}, ErrorCheck(err)
	}

	return tasksInfoList, nil
}

// This is a generic function to return a list of task information. Two types of clients us it: (a) users and (b) deployers.
// If the client is a user, the taskInfo may or may not include taskKey (as required).
// If the deployer calls it, the taskKey is likely to be returned.
func (db *MongoService) GetTasksInfoGeneric(client string, jobId string, limit int32,
	inclKey bool, isDeployer bool) ([]openapi.TaskInfo, error) {
	filter := bson.M{util.DBFieldJobId: jobId}

	// for user invoked calls client is the userId
	// for deployer invoked calls client is the computeId
	if !isDeployer {
		// first find a job; so that we can proceed to fetching the info of tasks
		// for deployer calls, it is assumed that the job check was previously performed. We can directly fetch tasks.
		// TODO verify that this job-exists check is handled correctly even for internal system calls
		zap.S().Infof("Get info of all tasks in a job %s owned by user: %s", jobId, client)

		if err := db.jobExists(client, jobId); err != nil {
			zap.S().Warnf("failed to find a matching job: %v", err)
			return nil, ErrorCheck(err)
		}
	} else {
		if client != "" {
			filter[util.DBFieldComputeId] = client
		}
	}
	cursor, err := db.taskCollection.Find(context.TODO(), filter)
	if err != nil {
		zap.S().Warnf("failed to fetch task info: %v", err)

		return nil, ErrorCheck(err)
	}

	defer cursor.Close(context.TODO())

	tasksInfoList := []openapi.TaskInfo{}
	for cursor.Next(context.TODO()) {
		var taskInfo openapi.TaskInfo
		if err = cursor.Decode(&taskInfo); err != nil {
			zap.S().Errorf("Failed to decode task info: %v", err)

			return nil, ErrorCheck(err)
		}

		if !inclKey {
			// if inclKey is not enabled,
			// strip off the key field in the data structure
			//
			// So, the bool parameter should be enabled only for internal calls
			taskInfo.Key = hiddenMessage
		}

		tasksInfoList = append(tasksInfoList, taskInfo)
	}

	return tasksInfoList, nil
}
