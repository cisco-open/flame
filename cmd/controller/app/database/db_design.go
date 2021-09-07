package database

import (
	"os"

	"wwwin-github.cisco.com/eti/fledge/pkg/openapi"
)

// CreateDesign - Create a new design template entry in the database.
func CreateDesign(userId string, info openapi.Design) error {
	return DB.CreateDesign(userId, info)
}

func GetDesigns(userId string, limit int32) ([]openapi.DesignInfo, error) {
	return DB.GetDesigns(userId, limit)
}

func GetDesign(userId string, designId string) (openapi.Design, error) {
	return DB.GetDesign(userId, designId)
}

func GetDesignSchema(userId string, designId string, version string) (openapi.DesignSchema, error) {
	return DB.GetDesignSchema(userId, designId, version)
}

func GetDesignSchemas(userId string, designId string) ([]openapi.DesignSchema, error) {
	return DB.GetDesignSchemas(userId, designId)
}

func CreateDesignSchema(userId string, designId string, info openapi.DesignSchema) error {
	return DB.CreateDesignSchema(userId, designId, info)
}

func UpdateDesignSchema(userId string, designId string, version string, info openapi.DesignSchema) error {
	return DB.UpdateDesignSchema(userId, designId, version, info)
}

func CreateDesignCode(userId string, designId string, fileName string, fileVer string, fileData *os.File) error {
	return DB.CreateDesignCode(userId, designId, fileName, fileVer, fileData)
}
