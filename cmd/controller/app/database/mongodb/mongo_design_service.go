package mongodb

import (
	"context"
	"fmt"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.uber.org/zap"

	"wwwin-github.cisco.com/eti/fledge/pkg/openapi"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

// CreateDesign create a new design entry in the database
func (db *MongoService) CreateDesign(userId string, design openapi.Design) error {
	result, err := db.designCollection.InsertOne(context.TODO(), design)

	if err != nil {
		err = ErrorCheck(err)
		zap.S().Errorf("Failed to create new design: %v", err)

		return err
	}

	Id, ok := result.InsertedID.(primitive.ObjectID)
	if !ok {
		err := fmt.Errorf("failed to type-assert primitive.ObjectID")
		zap.S().Debug(err)
		return err
	}

	design.Id = Id.Hex()
	db.updateDesignId(Id)

	zap.S().Debugf("New design for user: %s inserted with ID: %s", userId, design.Id)

	return nil
}

func (db *MongoService) updateDesignId(Id primitive.ObjectID) error {
	zap.S().Debugf("Updating design schema request for designId: %s", Id.Hex())

	var updatedDoc openapi.Design
	filter := bson.M{util.DBFieldMongoID: Id}
	update := bson.M{"$set": bson.M{"id": Id.Hex()}}

	err := db.designCollection.FindOneAndUpdate(context.TODO(), filter, update).Decode(&updatedDoc)
	if err != nil {
		zap.S().Errorf("Failed to update the design: %v", err)
		return ErrorCheck(err)
	}

	return nil
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

// GetDesign returns the details about the given design id
func (db *MongoService) GetDesign(userId string, designId string) (openapi.Design, error) {
	zap.S().Debugf("Get design information for user: %s | desginId: %s", userId, designId)

	var design openapi.Design

	filter := bson.M{util.DBFieldMongoID: ConvertToObjectID(designId), util.DBFieldUserId: userId}
	err := db.designCollection.FindOne(context.TODO(), filter).Decode(&design)
	if err != nil {
		err = ErrorCheck(err)
		zap.S().Errorf("Failed to fetch design template information: %v", err)
	}

	return design, err
}
