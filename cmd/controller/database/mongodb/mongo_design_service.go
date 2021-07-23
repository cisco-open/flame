package mongodb

import (
	"context"

	"go.mongodb.org/mongo-driver/bson"
	"go.uber.org/zap"
	"wwwin-github.cisco.com/eti/fledge/pkg/objects"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

// CreateDesign create a new design entry in the database
func (db *MongoService) CreateDesign(userId string, design objects.Design) error {
	result, err := db.designCollection.InsertOne(context.TODO(), design)
	zap.S().Debugf("mongodb new design request for userId: %v inserted with ID: %v", userId, result)
	if err != nil {
		err = ErrorCheck(err)
		zap.S().Errorf("error while creating new design. %v", err)
	}
	return err
}

// GetDesigns get a lists of all the designs created by the user
//todo update the method to implement a limit next based cursor
func (db *MongoService) GetDesigns(userId string, limit int32) ([]objects.DesignInfo, error) {
	zap.S().Debugf("mongodb get list of designs for user: %s | limit: %d", userId, limit)
	filter := bson.M{util.UserId: userId}
	cursor, err := db.designCollection.Find(context.TODO(), filter)

	if err != nil {
		err = ErrorCheck(err)
		zap.S().Errorf("error while getting list of designs. %v", err)
		return nil, err
	}

	defer cursor.Close(context.TODO())
	var infoList []objects.DesignInfo

	for cursor.Next(context.TODO()) {
		var d objects.DesignInfo
		if err = cursor.Decode(&d); err != nil {
			err = ErrorCheck(err)
			zap.S().Errorf("error while decoding design info. %v", err)
			return nil, err
		}
		infoList = append(infoList, d)
	}
	return infoList, err
}

// GetDesign get the details about the given design id
func (db *MongoService) GetDesign(userId string, designId string) (objects.Design, error) {
	zap.S().Debugf("mongodb get design information for user: %s | desginId: %s", userId, designId)
	var info objects.Design
	filter := bson.M{"_id": ConvertToObjectID(designId), util.UserId: userId}
	err := db.designCollection.FindOne(context.TODO(), filter).Decode(&info)
	if err != nil {
		err = ErrorCheck(err)
		zap.S().Errorf("error while fetching design template information. %v", err)
	}
	return info, err
}
