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
	"context"
	"fmt"
	"os"

	"go.mongodb.org/mongo-driver/bson"
	"go.uber.org/zap"

	"github.com/cisco-open/flame/pkg/openapi"
	"github.com/cisco-open/flame/pkg/util"
)

type FileInfo struct {
	ZipName  string          `json:"zipname"`
	DataSet  []util.FileData `json:"dataset"`
	Revision int32           `json:"revision"`
}

func (db *MongoService) CreateDesignCode(userId string, designId string, fileName string, fileData *os.File) error {
	zap.S().Debugf("Received CreateDesignCode POST request: %s | %s | %s", userId, designId, fileName)

	fdList, err := util.UnzipFile(fileData)
	if err != nil {
		zap.S().Errorf("Failed to read the file: %v", err)
		return err
	}

	fileInfo := FileInfo{
		ZipName: fileName,
		DataSet: fdList,
	}
	db.incrementCodeRevision(userId, designId, &fileInfo)

	var updatedDoc openapi.Design
	filter := bson.M{util.DBFieldUserId: userId, util.DBFieldId: designId}
	update := bson.M{"$set": bson.M{"code": fileInfo}}

	err = db.designCollection.FindOneAndUpdate(context.TODO(), filter, update).Decode(&updatedDoc)
	if err != nil {
		zap.S().Errorf("Failed to insert a design code: %v", err)
		return err
	}

	zap.S().Debugf("Created file %s successufully", fileName)

	return nil
}

func (db *MongoService) GetDesignCode(userId string, designId string) ([]byte, error) {
	fileInfo, err := db.getDesignCodeFileInfo(userId, designId)
	if err != nil {
		zap.S().Debugf("Failed to get code for user: %s, design: %s", userId, designId)
		return nil, err
	}

	zap.S().Debugf("Obtained code for user: %s, design: %s", userId, designId)
	return util.ZipFile(fileInfo.DataSet)
}

func (db *MongoService) DeleteDesignCode(userId string, designId string) error {
	zap.S().Debugf("delete design code: %v, %v", userId, designId)

	emptyFileInfo := FileInfo{}
	filter := bson.M{util.DBFieldUserId: userId, util.DBFieldId: designId}
	update := bson.M{"$set": bson.M{util.DBFieldCode: emptyFileInfo}}

	updateRes, err := db.designCollection.UpdateOne(context.TODO(), filter, update)
	if err != nil {
		return fmt.Errorf("failed to delete design code: %v", err)
	}

	if updateRes.ModifiedCount == 0 {
		return fmt.Errorf("code for design '%s' not found", designId)
	}

	zap.S().Debugf("Deleted design code for user: %s, design: %s", userId, designId)

	return nil
}

func (db *MongoService) GetDesignCodeRevision(userId string, designId string) (openapi.CodeApiResponse, error) {
	// match stage
	match := bson.M{
		"$match": bson.M{
			"userid": userId,
			"id":     designId,
		},
	}

	// project code
	project := bson.M{
		"$project": bson.M{
			"revision": "$code.revision",
		},
	}

	pipeline := []bson.M{match, project}
	fileInfo := FileInfo{}
	err := db.projectDesignCodeFileInfo(pipeline, &fileInfo)
	if err != nil {
		zap.S().Errorf("Failed to get design code info: %v", err)
		return openapi.CodeApiResponse{}, err
	}

	zap.S().Debugf("code revision number: %d", fileInfo.Revision)

	return openapi.CodeApiResponse{Revision: fileInfo.Revision}, nil
}

func (db *MongoService) getDesignCodeFileInfo(userId string, designId string) (FileInfo, error) {
	// match stage
	match := bson.M{
		"$match": bson.M{
			"userid": userId,
			"id":     designId,
		},
	}

	// project code
	project := bson.M{
		"$project": bson.M{
			"zipname":  "$code.zipname",
			"dataset":  "$code.dataset",
			"revision": "$code.revision",
		},
	}

	pipeline := []bson.M{match, project}
	updatedDoc := FileInfo{}
	err := db.projectDesignCodeFileInfo(pipeline, &updatedDoc)
	if err != nil {
		zap.S().Errorf("Failed to get design code: %v", err)
		return FileInfo{}, err
	}

	if len(updatedDoc.DataSet) == 0 {
		return FileInfo{}, fmt.Errorf("no design code found")
	}

	return updatedDoc, nil
}

func (db *MongoService) projectDesignCodeFileInfo(pipeline []bson.M, fileInfo *FileInfo) error {
	cursor, err := db.designCollection.Aggregate(context.TODO(), pipeline)
	if err != nil {
		zap.S().Errorf("Failed to fetch design code: %v", err)
		return err
	}

	defer func() {
		err = cursor.Close(context.TODO())
		if err != nil {
			zap.S().Warnf("Failed to close cursor: %v", err)
		}
	}()

	for cursor.Next(context.TODO()) {
		err = cursor.Decode(fileInfo)
		if err != nil {
			zap.S().Errorf("Failed to decode design code: %v", err)
			return err
		}
	}

	return nil
}

func (db *MongoService) incrementCodeRevision(userId string, designId string, fi *FileInfo) {
	curFileInfo, err := db.getDesignCodeFileInfo(userId, designId)
	if err != nil {
		fi.Revision = 1
	} else {
		fi.Revision = curFileInfo.Revision + 1
	}
}
