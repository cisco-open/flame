// Copyright (c) 2021 Cisco Systems, Inc. and its affiliates
// All rights reserved
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package mongodb

import (
	"archive/zip"
	"bytes"
	"context"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"

	"go.mongodb.org/mongo-driver/bson"
	"go.uber.org/zap"

	"wwwin-github.cisco.com/eti/fledge/pkg/openapi"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

type FileData struct {
	BaseName string `json:"basename"`
	FullName string `json:"fullname"`
	Data     string `json:"data"`
}

type FileInfo struct {
	ZipName string     `json:"zipname"`
	Version string     `json:"version"`
	DataSet []FileData `json:"dataset"`
}

func (db *MongoService) CreateDesignCode(userId string, designId string, fileName string, fileVer string, fileData *os.File) error {
	zap.S().Debugf("Received CreateDesignCode POST request: %s | %s | %s | %s", userId, designId, fileName, fileVer)

	fdList, err := unzipFile(fileData)
	if err != nil {
		zap.S().Errorf("Failed to read the file: %v", err)
		return err
	}

	curVer := db.GetLatestDesignCodeVersion(userId, designId)
	nextVer := IncrementVersion(curVer)
	fileInfo := FileInfo{
		ZipName: fileName,
		Version: nextVer,
		DataSet: fdList,
	}

	var updatedDoc openapi.Design
	filter := bson.M{util.DBFieldUserId: userId, util.DBFieldId: designId}
	update := bson.M{"$push": bson.M{"codes": fileInfo}}

	err = db.designCollection.FindOneAndUpdate(context.TODO(), filter, update).Decode(&updatedDoc)
	if err != nil {
		zap.S().Errorf("Failed to insert a design code: %v", err)
		return err
	}

	zap.S().Debugf("Created file %s successufully", fileName)

	return nil
}

func (db *MongoService) GetLatestDesignCodeVersion(userId string, designId string) string {
	// match stage
	stage1 := bson.M{
		"$match": bson.M{
			"userid": userId,
			"id":     designId,
		},
	}

	// unwind stage
	stage2 := bson.M{"$unwind": "$codes"}

	// choose last code
	stage3 := bson.M{
		"$group": bson.M{
			"_id": "$_id",
			"codes": bson.M{
				"$last": "$codes",
			},
		},
	}

	// project zipname and version
	stage4 := bson.M{
		"$project": bson.M{
			"zipname": "$codes.zipname",
			"version": "$codes.version",
		},
	}

	pipeline := []bson.M{stage1, stage2, stage3, stage4}

	cursor, err := db.designCollection.Aggregate(context.TODO(), pipeline)
	if err != nil {
		zap.S().Errorf("Failed to fetch design code info: %v", err)
		return unknownVersion
	}

	defer func() {
		err = cursor.Close(context.TODO())
		if err != nil {
			zap.S().Warnf("Failed to close cursor: %v", err)
		}
	}()

	updatedDoc := FileInfo{}
	for cursor.Next(context.TODO()) {
		err = cursor.Decode(&updatedDoc)
		if err != nil {
			zap.S().Errorf("Failed to decode design code info: %v", err)
			return unknownVersion
		}
	}

	return updatedDoc.Version
}

func (db *MongoService) GetDesignCode(userId string, designId string, fileVer string) ([]byte, error) {
	// match stage with a specific version
	stage1 := bson.M{
		"$match": bson.M{
			"userid":        userId,
			"id":            designId,
			"codes.version": fileVer,
		},
	}

	// unwind stage
	stage2 := bson.M{"$unwind": "$codes"}

	// stage3 unneeded for specific version
	var stage3 *bson.M

	// project zipname, version and dataset
	stage4 := bson.M{
		"$project": bson.M{
			"zipname": "$codes.zipname",
			"version": "$codes.version",
			"dataset": "$codes.dataset",
		},
	}

	if fileVer == latestVersion {
		// for latest version, match only with userId and designId
		stage1 = bson.M{
			"$match": bson.M{
				"userid": userId,
				"id":     designId,
			},
		}

		// choose last code
		stage3 = &bson.M{
			"$group": bson.M{
				"_id": "$_id",
				"codes": bson.M{
					"$last": "$codes",
				},
			},
		}
	}

	pipeline := []bson.M{stage1, stage2}
	if stage3 != nil {
		pipeline = append(pipeline, *stage3)
	}
	pipeline = append(pipeline, stage4)

	cursor, err := db.designCollection.Aggregate(context.TODO(), pipeline)
	if err != nil {
		zap.S().Errorf("Failed to fetch design code: %v", err)
		return nil, err
	}

	defer func() {
		err = cursor.Close(context.TODO())
		if err != nil {
			zap.S().Warnf("Failed to close cursor: %v", err)
		}
	}()

	updatedDoc := FileInfo{}
	for cursor.Next(context.TODO()) {
		err = cursor.Decode(&updatedDoc)
		if err != nil {
			zap.S().Errorf("Failed to decode design code: %v", err)
			return nil, err
		}
	}

	if len(updatedDoc.DataSet) == 0 {
		return nil, fmt.Errorf("no design code found")
	}

	return zipFile(updatedDoc.DataSet)
}

func unzipFile(file *os.File) ([]FileData, error) {
	reader, err := zip.OpenReader(file.Name())
	if err != nil {
		return nil, err
	}
	defer reader.Close()

	fdList := []FileData{}
	for _, f := range reader.File {
		rc, err := f.Open()
		if err != nil {
			return nil, err
		}

		if f.FileInfo().IsDir() {
			// No need to store directory information
			rc.Close()
			continue
		}

		data, err := ioutil.ReadAll(rc)
		if err != nil {
			rc.Close()
			return nil, err
		}

		fd := FileData{
			BaseName: filepath.Base(f.Name),
			FullName: f.Name,
			Data:     string(data),
		}

		fdList = append(fdList, fd)

		rc.Close()
	}

	if len(fdList) == 0 {
		return nil, fmt.Errorf("no file found in %s", file.Name())
	}

	return fdList, nil
}

func zipFile(fdList []FileData) ([]byte, error) {
	// Create a buffer to write an archive to
	buf := new(bytes.Buffer)

	// Create a new zip archive
	writer := zip.NewWriter(buf)
	for _, fd := range fdList {
		f, err := writer.Create(fd.FullName)
		if err != nil {
			return nil, err
		}

		_, err = f.Write([]byte(fd.Data))
		if err != nil {
			return nil, err
		}
	}

	// Make sure to check the error on Close
	err := writer.Close()
	if err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}
