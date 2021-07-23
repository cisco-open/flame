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

	filter := bson.M{"_id": ConvertToObjectID(designId), util.UserId: userId}
	opts := options.FindOne().SetProjection(bson.M{"schemas": 1})

	//selecting a specific schema from list of schemas
	if getType == util.ID {
		filter = bson.M{"_id": ConvertToObjectID(designId), "schemas._id": ConvertToObjectID(schemaId)}
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

// UpdateDesignSchema adds schema design to the design template information
func (db *MongoService) UpdateDesignSchema(userId string, designId string, ds objects.DesignSchema) error {
	zap.S().Debugf("mongodb update design schema request for userId:%s | designId:%s", userId, designId)

	ds.ID = primitive.NewObjectID()

	var updatedDoc objects.Design
	filter := bson.M{"_id": ConvertToObjectID(designId), util.UserId: userId}
	update := bson.M{"$push": bson.M{"schemas": ds}}
	err := db.designCollection.FindOneAndUpdate(context.TODO(), filter, update).Decode(&updatedDoc)

	if err != nil {
		zap.S().Errorf("error while updating the design template. %v", err)
		err = ErrorCheck(err)
	}
	return err
}
