package mongodb

import (
	"errors"

	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
	"go.uber.org/zap"
)

/*
	Tutorial - https://www.mongodb.com/blog/post/quick-start-golang--mongodb--how-to-read-documents

	Bson.M vs .D https://stackoverflow.com/questions/64281675/bson-d-vs-bson-m-for-find-queries
*/
func ConvertToObjectID(id string) primitive.ObjectID {
	objID, err := primitive.ObjectIDFromHex(id)
	if err != nil {
		zap.S().Fatal(err)
	}
	return objID
}

func ErrorCheck(err error) error {
	if mongo.IsDuplicateKeyError(err) {
		err = errors.New("duplicate key error")
	} else if mongo.IsNetworkError(err) {
		err = errors.New("db connection error")
	} else if mongo.IsTimeout(err) {
		err = errors.New("db connection timeout error")
	} else if err == mongo.ErrNoDocuments {
		//ErrNoDocuments means that the filter did not match any documents in the collection
		err = errors.New("no document found")
	}

	return err
}
