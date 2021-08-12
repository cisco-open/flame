package database

import (
	"wwwin-github.cisco.com/eti/fledge/pkg/objects"
)

// CreateDesign - Create a new design template entry in the database.
func CreateDesign(userId string, info objects.Design) error {
	return DB.CreateDesign(userId, info)
}

func GetDesigns(userId string, limit int32) ([]objects.DesignInfo, error) {
	return DB.GetDesigns(userId, limit)
}

func GetDesign(userId string, designId string) (objects.Design, error) {
	return DB.GetDesign(userId, designId)
}

func GetDesignSchema(userId string, designId string, getType string, schemaId string) ([]objects.DesignSchema, error) {
	return DB.GetDesignSchema(userId, designId, getType, schemaId)
}

func CreateDesignSchema(userId string, designId string, info objects.DesignSchema) error {
	return DB.CreateDesignSchema(userId, designId, info)
}

func UpdateDesignSchema(userId string, designId string, info objects.DesignSchema) error {
	return DB.UpdateDesignSchema(userId, designId, info)
}
