package mongodb

import (
	"context"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo/options"
	"go.uber.org/zap"

	"wwwin-github.cisco.com/eti/fledge/pkg/openapi"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

func (db *MongoService) GetDesignSchema(userId string, designId string, getType string, schemaId string) ([]openapi.DesignSchema, error) {
	zap.S().Debugf("Getting schema details for userId: %s | designId: %s | getType: %s | schemaId: %s",
		userId, designId, getType, schemaId)

	filter := bson.M{util.DBFieldMongoID: ConvertToObjectID(designId)}
	opts := options.FindOne().SetProjection(bson.M{"schemas": 1})

	// selecting a specific schema from list of schemas
	if getType == util.ID {
		filter = bson.M{util.DBFieldMongoID: ConvertToObjectID(designId), "schemas.id": schemaId}
		opts = options.FindOne().SetProjection(bson.M{"schemas.$": 1})
	}

	designSchemas := openapi.DesignSchemas{}
	err := db.designCollection.FindOne(context.TODO(), filter, opts).Decode(&designSchemas)
	if err != nil {
		err = ErrorCheck(err)
		zap.S().Errorf("Failed to fetch design schema information: %v", err)
		return []openapi.DesignSchema{}, err
	}

	zap.S().Debug(designSchemas)

	return designSchemas.Schemas, err
}

// CreateDesignSchema adds schema design to the design template information
func (db *MongoService) CreateDesignSchema(userId string, designId string, ds openapi.DesignSchema) error {
	zap.S().Debugf("Creating design schema request for userId: %s | designId: %s", userId, designId)

	ds.Id = primitive.NewObjectID().Hex()

	var updatedDoc openapi.Design
	filter := bson.M{util.DBFieldMongoID: ConvertToObjectID(designId), util.DBFieldUserId: userId}
	update := bson.M{"$push": bson.M{"schemas": ds}}

	err := db.designCollection.FindOneAndUpdate(context.TODO(), filter, update).Decode(&updatedDoc)
	if err != nil {
		zap.S().Errorf("Failed to insert the design template. %v", err)
		return ErrorCheck(err)
	}

	return nil
}

func (db *MongoService) UpdateDesignSchema(userId string, designId string, ds openapi.DesignSchema) error {
	zap.S().Debugf("Updating design schema request for userId: %s | designId: %s| schemaId: %s", userId, designId, ds.Id)

	var updatedDoc openapi.Design
	filter := bson.M{util.DBFieldMongoID: ConvertToObjectID(designId), "schemas.id": ds.Id}
	update := bson.M{"$set": bson.M{"schemas.$": ds}}

	err := db.designCollection.FindOneAndUpdate(context.TODO(), filter, update).Decode(&updatedDoc)
	if err != nil {
		zap.S().Errorf("Failed to update the design template: %v", err)
		return ErrorCheck(err)
	}

	return nil
}
