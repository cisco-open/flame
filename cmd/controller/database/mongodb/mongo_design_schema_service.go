package mongodb

import (
	"context"

	"go.mongodb.org/mongo-driver/mongo/options"
	"wwwin-github.cisco.com/eti/fledge/pkg/objects"

	"wwwin-github.cisco.com/eti/fledge/pkg/util"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.uber.org/zap"
)

func (db *MongoService) GetDesignSchema(userId string, designId string, getType string, schemaId string) ([]objects.DesignSchema, error) {
	zap.S().Debugf("mongodb get schema details for userId:%s | designId:%s | getType:%s | schemaId:%s", userId, designId, getType, schemaId)

	filter := bson.M{util.DBFieldMongoID: ConvertToObjectID(designId)}
	opts := options.FindOne().SetProjection(bson.M{"schemas": 1})

	//selecting a specific schema from list of schemas
	if getType == util.ID {
		filter = bson.M{util.DBFieldMongoID: ConvertToObjectID(designId), "schemas._id": schemaId}
		opts = options.FindOne().SetProjection(bson.M{"schemas.$": 1})
	}

	var info objects.DesignSchemas
	err := db.designCollection.FindOne(context.TODO(), filter, opts).Decode(&info)

	if err != nil {
		err = ErrorCheck(err)
		zap.S().Errorf("error while fetching design schema information. %v", err)
		return []objects.DesignSchema{}, err
	}
	return info.Schemas, err
}

// CreateDesignSchema adds schema design to the design template information
func (db *MongoService) CreateDesignSchema(userId string, designId string, ds objects.DesignSchema) error {
	zap.S().Debugf("mongodb create design schema request for userId:%s | designId:%s", userId, designId)

	ds.ID = GetStringID(primitive.NewObjectID())
	var updatedDoc objects.Design
	filter := bson.M{util.DBFieldMongoID: ConvertToObjectID(designId), util.DBFieldUserId: userId}
	update := bson.M{"$push": bson.M{"schemas": ds}}

	err := db.designCollection.FindOneAndUpdate(context.TODO(), filter, update).Decode(&updatedDoc)
	if err != nil {
		zap.S().Errorf("error while inserting the design template. %v", err)
		return ErrorCheck(err)
	}
	return nil
}

func (db *MongoService) UpdateDesignSchema(userId string, designId string, ds objects.DesignSchema) error {
	zap.S().Debugf("mongodb update design schema request for userId:%s | designId:%s| schemaId: %s", userId, designId, ds.ID)

	var updatedDoc objects.Design
	filter := bson.M{util.DBFieldMongoID: ConvertToObjectID(designId), "schemas._id": ds.ID}
	update := bson.M{"$set": bson.M{"schemas.$": ds}}

	err := db.designCollection.FindOneAndUpdate(context.TODO(), filter, update).Decode(&updatedDoc)
	if err != nil {
		zap.S().Errorf("error while updating the design template. %v", err)
		return ErrorCheck(err)
	}
	return nil
}
