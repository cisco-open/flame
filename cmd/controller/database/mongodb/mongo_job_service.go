package mongodb

import (
	"context"
	"errors"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.uber.org/zap"

	"wwwin-github.cisco.com/eti/fledge/pkg/objects"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

// SubmitJob creates a new job and return the jobId which is used by the controller to inform the agent about new job.
func (db *MongoService) SubmitJob(userId string, info objects.JobInfo) (string, error) {
	t := time.Now()
	info.Timestamp = objects.JobInfoTimestamp{
		CreatedAt:   t.Unix(),
		StartedAt:   0,
		UpdatedAt:   0,
		CompletedAt: 0,
	}
	info.Status = util.Initializing
	result, err := db.jobCollection.InsertOne(context.TODO(), info)
	zap.S().Debugf("mongodb new job request for userId: %v inserted with ID: %v", userId, result.InsertedID)
	if err != nil {
		err = ErrorCheck(err)
		zap.S().Errorf("error while submitting new job request. %v", err)
	}
	return GetStringID(result.InsertedID), err
}

func (db *MongoService) GetJob(userId string, jobId string) (objects.JobInfo, error) {
	zap.S().Debugf("mongodb get job for userId: %s with jobId: %s", userId, jobId)
	filter := bson.M{util.UserId: userId, util.MongoID: ConvertToObjectID(jobId)}
	var info objects.JobInfo
	err := db.jobCollection.FindOne(context.TODO(), filter).Decode(&info)
	if err != nil {
		err = ErrorCheck(err)
		zap.S().Errorf("error while fetching job details. %v", err)
	}
	return info, nil
}
func (db *MongoService) GetJobs(userId string, getType string, designId string, limit int32) ([]objects.JobInfo, error) {
	zap.S().Debugf("mongodb get jobs detail for userId: %s | | getType: %s | designId: %s ", userId, getType, designId)
	filter := bson.M{util.UserId: userId}
	if getType == util.Design {
		filter = bson.M{util.UserId: userId, util.DesignId: designId}
	}

	cursor, err := db.jobCollection.Find(context.TODO(), filter)

	if err != nil {
		err = ErrorCheck(err)
		zap.S().Errorf("error while getting jobs list. %v", err)
		return nil, err
	}

	defer cursor.Close(context.TODO())
	var infoList []objects.JobInfo
	for cursor.Next(context.TODO()) {
		var d objects.JobInfo
		if err = cursor.Decode(&d); err != nil {
			err = ErrorCheck(err)
			zap.S().Errorf("error while decoding job info. %v", err)
			return nil, err
		}
		infoList = append(infoList, d)
	}
	return infoList, nil
}

func (db *MongoService) UpdateJob(userId string, jobId string) (objects.JobInfo, error) {
	return objects.JobInfo{}, nil
}
func (db *MongoService) DeleteJob(userId string, jobId string) error {
	return nil
}

/*
	Internal Functions to be used by Controller only
    - - - - - - - -- - - -- - - -- - - -- - - -- - - -
*/

func (db *MongoService) UpdateJobDetails(jobId string, updateType string, msg []byte) error {
	if updateType == util.AddJobNodes {
		return db.addJobNodes(jobId, msg)
	}
	return errors.New("update job details request failed due to invalid update type")
}

func (db *MongoService) addJobNodes(jobId string, msg []byte) error {
	var nodesInfo []objects.ServerInfo
	err := util.ByteToStruct(msg, &nodesInfo)
	if err != nil {
		return err
	}

	filter := bson.M{util.MongoID: ConvertToObjectID(jobId)}
	update := bson.M{"$set": bson.M{"nodes": nodesInfo}}
	var updatedDocument bson.M
	err = db.jobCollection.FindOneAndUpdate(context.TODO(), filter, update).Decode(&updatedDocument)
	if err != nil {
		zap.S().Errorf("error while updating the design template. %v", err)
		err = ErrorCheck(err)
	}
	return err
}
