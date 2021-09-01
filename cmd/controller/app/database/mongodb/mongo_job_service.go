package mongodb

import (
	"context"
	"errors"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.uber.org/zap"

	"wwwin-github.cisco.com/eti/fledge/pkg/openapi"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

// SubmitJob creates a new job and return the jobId which is used by the controller to inform the fledgelet about new job.
func (db *MongoService) SubmitJob(userId string, info openapi.JobInfo) (string, error) {
	t := time.Now()
	info.Timestamp = openapi.JobInfoTimestamp{
		CreatedAt:   t.Unix(),
		StartedAt:   0,
		UpdatedAt:   0,
		CompletedAt: 0,
	}
	info.Status = util.InitState
	result, err := db.jobCollection.InsertOne(context.TODO(), info)
	zap.S().Debugf("mongodb new job request for userId: %v inserted with ID: %v", userId, result.InsertedID)
	if err != nil {
		err = ErrorCheck(err)
		zap.S().Errorf("error while submitting new job request. %v", err)
	}
	return GetStringID(result.InsertedID), err
}

func (db *MongoService) GetJob(userId string, jobId string) (openapi.JobInfo, error) {
	zap.S().Debugf("mongodb get job for userId: %s with jobId: %s", userId, jobId)
	filter := bson.M{util.DBFieldMongoID: ConvertToObjectID(jobId)}
	var info openapi.JobInfo
	err := db.jobCollection.FindOne(context.TODO(), filter).Decode(&info)
	if err != nil {
		err = ErrorCheck(err)
		zap.S().Errorf("error while fetching job details. %v", err)
	}
	return info, nil
}
func (db *MongoService) GetJobs(userId string, getType string, designId string, limit int32) ([]openapi.JobInfo, error) {
	zap.S().Debugf("mongodb get jobs detail for userId: %s | | getType: %s | designId: %s ", userId, getType, designId)
	filter := bson.M{util.DBFieldUserId: userId}
	if getType == util.Design {
		filter = bson.M{util.DBFieldUserId: userId, util.DBFieldDesignId: designId}
	}
	return db.getJobsInfo(filter)
}

//func (db *MongoService) GetJobsDetailsBy(userId string, getType string, in map[string]string) ([]openapi.JobInfo, error) {
//	if getType == util.GetBySchemaId {
//		filter := bson.M{util.DBFieldDesignId: in[util.DBFieldDesignId], util.DBFieldSchemaId: in[util.DBFieldSchemaId]}
//		return db.getJobsInfo(filter)
//	}
//	return nil, nil
//}

func (db *MongoService) getJobsInfo(filter primitive.M) ([]openapi.JobInfo, error) {
	cursor, err := db.jobCollection.Find(context.TODO(), filter)

	if err != nil {
		err = ErrorCheck(err)
		zap.S().Errorf("error while getting jobs list. %v", err)
		return nil, err
	}

	defer cursor.Close(context.TODO())
	var infoList []openapi.JobInfo
	for cursor.Next(context.TODO()) {
		var d openapi.JobInfo
		if err = cursor.Decode(&d); err != nil {
			err = ErrorCheck(err)
			zap.S().Errorf("error while decoding job info. %v", err)
			return nil, err
		}
		infoList = append(infoList, d)
	}
	return infoList, nil
}

func (db *MongoService) UpdateJob(userId string, jobId string) (openapi.JobInfo, error) {
	return openapi.JobInfo{}, nil
}
func (db *MongoService) DeleteJob(userId string, jobId string) error {
	return nil
}

/*
	Internal Functions to be used by Controller only
    - - - - - - - -- - - -- - - -- - - -- - - -- - - -
*/
func (db *MongoService) UpdateJobDetails(jobId string, updateType string, msg interface{}) error {
	switch updateType {
	case util.AddJobNodes:
		return db.addJobNodes(jobId, msg)
	case util.JobStatus:
		return db.updateNodeJobStatus(jobId, msg)
	case util.ChangeJobSchema:
		return db.changeJobSchema(jobId, msg)
	default:
		return errors.New("update job details request failed due to invalid update type")
	}
}

func (db *MongoService) addJobNodes(jobId string, msg interface{}) error {
	nodesInfo := msg.([]openapi.ServerInfo)
	zap.S().Debugf("nodesInfo : %v", nodesInfo)

	filter := bson.M{util.DBFieldMongoID: ConvertToObjectID(jobId)}
	update := bson.M{"$push": bson.M{"nodes": bson.M{"$each": nodesInfo}}}
	var updatedDocument bson.M
	err := db.jobCollection.FindOneAndUpdate(context.TODO(), filter, update).Decode(&updatedDocument)
	if err != nil {
		zap.S().Errorf("error while adding new job nodes. %v", err)
		return ErrorCheck(err)
	}
	return nil
}

func (db *MongoService) updateNodeJobStatus(jobId string, msg interface{}) error {
	info := msg.(map[string]string)
	zap.S().Debugf("Updating the job %s state for the agent. %v", jobId, info)

	filter := bson.M{util.DBFieldMongoID: ConvertToObjectID(jobId), "nodes.uuid": info[util.ID]}
	update := bson.M{"$set": bson.M{"nodes.$.state": info[util.State]}}
	var updatedDocument bson.M
	err := db.jobCollection.FindOneAndUpdate(context.TODO(), filter, update).Decode(&updatedDocument)
	if err != nil {
		zap.S().Errorf("error while updating the node job state. %v", err)
		return ErrorCheck(err)
	}
	return nil
}

//changeJobSchema when schema is changed for the existing job new nodes might be added to the job.
//This method update both the schema id and adds new nodes, if created for the existing job).
func (db *MongoService) changeJobSchema(jobId string, msg interface{}) error {
	info := msg.(map[string]interface{})
	zap.S().Debugf("change job design schema. %v", info)

	newSchemaId := info[util.DBFieldSchemaId].(string)
	newNodes := info[util.DBFieldNodes].([]openapi.ServerInfo)

	filter := bson.M{util.DBFieldMongoID: ConvertToObjectID(jobId)}
	update := bson.M{"$set": bson.M{util.DBFieldSchemaId: newSchemaId}}
	if len(newNodes) != 0 {
		update = bson.M{"$set": bson.M{util.DBFieldSchemaId: newSchemaId}, "$push": bson.M{"nodes": bson.M{"$each": newNodes}}}
	}

	var updatedDocument bson.M
	err := db.jobCollection.FindOneAndUpdate(context.TODO(), filter, update).Decode(&updatedDocument)
	if err != nil {
		zap.S().Errorf("error while updating the schema for the given job. %v", err)
		return ErrorCheck(err)
	}
	return nil
}
