package database

import (
	objects2 "wwwin-github.cisco.com/fledge/fledge/pkg/objects"
)

// StoreCollection provides collection of all the db stores in the application.
type StoreCollection interface {
	DesignStore
}

// DesignStore is the collection of db APIs related to the designs
type DesignStore interface {
	CreateDesign(userId string, info objects2.Design) error

	GetDesigns(userId string, limit int32) ([]objects2.DesignInfo, error)
	GetDesign(userId string, designId string) (objects2.Design, error)

	GetDesignSchema(userId string, designId string, getType string, schemaId string) ([]objects2.DesignSchema, error)
	UpdateDesignSchema(userId string, designId string, info objects2.DesignSchema) error
}
