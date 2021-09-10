package mongodb

import (
	"context"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.uber.org/zap"

	"wwwin-github.cisco.com/eti/fledge/pkg/openapi"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

// CreateDataset creates a new dataset entry in the database
func (db *MongoService) CreateDataset(userId string, dataset openapi.DatasetInfo) error {
	// override user in the DatasetInfo struct to prevent wrong user name is recorded in the db
	dataset.UserId = userId

	result, err := db.datasetCollection.InsertOne(context.TODO(), dataset)
	if err != nil {
		err = ErrorCheck(err)
		zap.S().Errorf("Failed to create new dataset: %v", err)

		return err
	}

	err = db.setDatasetId(result.InsertedID.(primitive.ObjectID))
	if err != nil {
		zap.S().Errorf("Failed to set Id: %v", err)
		return err
	}

	zap.S().Debugf("New dataset for user: %s inserted with ID: %s", userId, dataset.Id)

	return nil
}

func (db *MongoService) setDatasetId(docId primitive.ObjectID) error {
	filter := bson.M{util.DBFieldMongoID: docId}
	update := bson.M{"$set": bson.M{util.DBFieldId: GetStringID(docId)}}

	updatedDoc := openapi.DatasetInfo{}
	err := db.datasetCollection.FindOneAndUpdate(context.TODO(), filter, update).Decode(&updatedDoc)
	if err != nil {
		return ErrorCheck(err)
	}

	return nil
}
