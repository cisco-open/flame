package database

import (
	objects2 "wwwin-github.cisco.com/eti/fledge/pkg/objects"
)

// CreateDesign - Create a new design template entry in the database.
func CreateDesign(userId string, info objects2.Design) error {
	return DB.CreateDesign(userId, info)
}

func GetDesigns(userId string, limit int32) ([]objects2.DesignInfo, error) {
	return DB.GetDesigns(userId, limit)
}

func GetDesign(userId string, designId string) (objects2.Design, error) {
	return DB.GetDesign(userId, designId)
}

func GetDesignSchema(userId string, designId string, getType string, schemaId string) ([]objects2.DesignSchema, error) {
	return DB.GetDesignSchema(userId, designId, getType, schemaId)
}

func UpdateDesignSchema(userId string, designId string, info objects2.DesignSchema) error {
	return DB.UpdateDesignSchema(userId, designId, info)
}
