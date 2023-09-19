// Copyright 2022 Cisco Systems, Inc. and its affiliates
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

package mongodb

import (
	"errors"

	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
	"go.uber.org/zap"
)

const (
	updatedAt = "updatedat"
)

/*
Tutorial - https://www.mongodb.com/blog/post/quick-start-golang--mongodb--how-to-read-documents

Bson.M vs .D https://stackoverflow.com/questions/64281675/bson-d-vs-bson-m-for-find-queries
*/
func ConvertToObjectID(id string) primitive.ObjectID {
	objID, err := primitive.ObjectIDFromHex(id)
	if err != nil {
		zap.S().Errorf("error converting string to mongodbId. %v", err)
	}
	return objID
}

func GetStringID(id interface{}) string {
	return id.(primitive.ObjectID).Hex()
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
