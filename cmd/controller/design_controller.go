package controller

import (
	database2 "wwwin-github.cisco.com/fledge/fledge/cmd/controller/database"
	objects2 "wwwin-github.cisco.com/fledge/fledge/pkg/objects"
)

type DesignController interface {
	CreateNewDesign(userId string, info objects2.Design) error

	GetDesigns(userId string, limit int64) ([]objects2.DesignInfo, error)
	GetDesign(userId string, designId string) (objects2.Design, error)

	GetDesignSchema(userId string, designId string, getType string, schemaId string) (objects2.DesignSchemas, error)
	UpdateDesignSchema(userId string, designId string, info objects2.DesignSchema) error
}

func CreateNewDesign(userId string, info objects2.Design) error {
	//todo input validation
	return database2.CreateDesign(userId, info)
}

func GetDesigns(userId string, limit int32) ([]objects2.DesignInfo, error) {
	//todo input validation
	return database2.GetDesigns(userId, limit)
}

func GetDesign(userId string, designId string) (objects2.Design, error) {
	//todo input validation
	return database2.GetDesign(userId, designId)
}

func UpdateDesignSchema(userId string, designId string, ds objects2.DesignSchema) error {
	//todo input validation
	return database2.UpdateDesignSchema(userId, designId, ds)
}

func GetDesignSchema(userId string, designId string, getType string, schemaId string) ([]objects2.DesignSchema, error) {
	return database2.GetDesignSchema(userId, designId, getType, schemaId)
}
