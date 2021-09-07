package mongodb

import (
	"archive/zip"
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
	BaseName string
	FullName string
	Data     string
}

type FileInfo struct {
	ZipName string
	Version string
	DataSet []FileData
}

func (db *MongoService) CreateDesignCode(userId string, designId string, fileName string, fileVer string, fileData *os.File) error {
	zap.S().Debugf("Received CreateDesignCode POST request: %s | %s | %s | %s", userId, designId, fileName, fileVer)

	// TODO: version check and increment version

	fdList, err := unzip(fileData)
	if err != nil {
		zap.S().Errorf("Failed to read the file: %v", err)
		return err
	}

	fileInfo := FileInfo{
		ZipName: fileName,
		Version: fileVer,
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

func (db *MongoService) GetDesignCode(userId string, designId string, fileVer string) (*os.File, error) {
	if fileVer == latestVersion {
		// do something
		zap.S().Debugf("Looking for latest version")
	} else {
		// do something
		zap.S().Debugf("Looking for specific version")
	}

	return nil, fmt.Errorf("not implemented")
}

func unzip(file *os.File) ([]FileData, error) {
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
