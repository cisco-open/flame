package mongodb

import (
	"context"
	"fmt"
	"strconv"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo/options"
	"go.uber.org/zap"

	"wwwin-github.cisco.com/eti/fledge/pkg/openapi"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

const (
	latestVersion = "latest"
)

// CreateDesignSchema adds schema design to the design template information
func (db *MongoService) CreateDesignSchema(userId string, designId string, ds openapi.DesignSchema) error {
	zap.S().Debugf("Creating design schema request for userId: %s | designId: %s", userId, designId)

	latestSchema, err := db.GetDesignSchema(userId, designId, latestVersion)
	if err != nil {
		ds.Version = "1"
	} else {
		ds.Version = IncrementVersion(latestSchema.Version)
	}

	var updatedDoc openapi.Design
	filter := bson.M{util.DBFieldUserId: userId, util.DBFieldId: designId}
	update := bson.M{"$push": bson.M{"schemas": ds}}

	err = db.designCollection.FindOneAndUpdate(context.TODO(), filter, update).Decode(&updatedDoc)
	if err != nil {
		zap.S().Errorf("Failed to insert the design template. %v", err)
		return ErrorCheck(err)
	}

	return nil
}

func (db *MongoService) GetDesignSchema(userId string, designId string, version string) (openapi.DesignSchema, error) {
	zap.S().Debugf("Getting schema details for userId: %s | designId: %s | schema version: %s",
		userId, designId, version)

	filter := bson.M{util.DBFieldUserId: userId, util.DBFieldId: designId, "schemas.version": version}
	opts := options.FindOne().SetProjection(bson.M{"schemas.$": 1})

	if version == latestVersion {
		filter = bson.M{util.DBFieldUserId: userId, util.DBFieldId: designId}
		projection := bson.M{"schemas": bson.M{"$slice": -1}}
		opts = options.FindOne().SetProjection(projection)
	}

	updatedDoc := openapi.Design{}
	err := db.designCollection.FindOne(context.TODO(), filter, opts).Decode(&updatedDoc)
	if err != nil {
		err = ErrorCheck(err)
		zap.S().Errorf("Failed to fetch design schema information: %v", err)
		return openapi.DesignSchema{}, err
	}

	if len(updatedDoc.Schemas) == 0 {
		return openapi.DesignSchema{}, fmt.Errorf("no design schema found")
	}

	return updatedDoc.Schemas[0], err
}

func (db *MongoService) GetDesignSchemas(userId string, designId string) ([]openapi.DesignSchema, error) {
	zap.S().Debugf("Getting schema details for userId: %s | designId: %s", userId, designId)

	filter := bson.M{util.DBFieldUserId: userId, util.DBFieldId: designId}
	opts := options.FindOne().SetProjection(bson.M{"schemas": 1})

	updatedDoc := openapi.Design{}
	err := db.designCollection.FindOne(context.TODO(), filter, opts).Decode(&updatedDoc)
	if err != nil {
		err = ErrorCheck(err)
		zap.S().Errorf("Failed to fetch design schema information: %v", err)
		return []openapi.DesignSchema{}, err
	}

	zap.S().Debug(updatedDoc.Schemas)

	return updatedDoc.Schemas, err
}

func (db *MongoService) UpdateDesignSchema(userId string, designId string, ds openapi.DesignSchema) error {
	zap.S().Debugf("Updating design schema request for userId: %s | designId: %s| version: %s", userId, designId, ds.Version)

	filter := bson.M{util.DBFieldUserId: userId, util.DBFieldId: designId, "schemas.version": ds.Version}
	update := bson.M{"$set": bson.M{"schemas.$": ds}}

	var updatedDoc openapi.Design
	err := db.designCollection.FindOneAndUpdate(context.TODO(), filter, update).Decode(&updatedDoc)
	if err != nil {
		zap.S().Errorf("Failed to update the design template: %v", err)
		return ErrorCheck(err)
	}

	return nil
}

func IncrementVersion(version string) string {
	num, err := strconv.Atoi(version)
	if err != nil {
		return "1"
	}

	return fmt.Sprintf("%d", num+1)
}
