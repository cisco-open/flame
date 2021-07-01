package controllerapi

import (
	"wwwin-github.cisco.com/eti/fledge/cmd/controller/database"
	"wwwin-github.cisco.com/eti/fledge/pkg/objects"
)

type DesignController struct{}

func (i *DesignController) CreateNewDesign(userId string, info objects.Design) error {

	return database.CreateDesign(userId, info)
}

func (i *DesignController) GetDesigns(userId string, limit int32) ([]objects.DesignInfo, error) {
	return database.GetDesigns(userId, limit)
}

func (i *DesignController) GetDesign(userId string, designId string) (objects.Design, error) {
	return database.GetDesign(userId, designId)
}

func (i *DesignController) UpdateDesignSchema(userId string, designId string, ds objects.DesignSchema) error {
	return database.UpdateDesignSchema(userId, designId, ds)
}

func (i *DesignController) GetDesignSchema(userId string, designId string, getType string, schemaId string) ([]objects.DesignSchema, error) {
	return database.GetDesignSchema(userId, designId, getType, schemaId)
}
