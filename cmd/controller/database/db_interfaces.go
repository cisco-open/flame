package database

import (
	"wwwin-github.cisco.com/eti/fledge/pkg/objects"
)

// StoreCollection provides collection of all the db stores in the application.
type StoreCollection interface {
	DesignStore
	JobStore
}

// DesignStore is the collection of db APIs related to the designs
type DesignStore interface {
	CreateDesign(userId string, info objects.Design) error

	GetDesigns(userId string, limit int32) ([]objects.DesignInfo, error)
	GetDesign(userId string, designId string) (objects.Design, error)

	GetDesignSchema(userId string, designId string, getType string, schemaId string) ([]objects.DesignSchema, error)
	UpdateDesignSchema(userId string, designId string, info objects.DesignSchema) error
}

type JobStore interface {
	SubmitJob(userId string, info objects.JobInfo) (string, error)

	GetJob(userId string, jobId string) (objects.JobInfo, error)
	GetJobs(userId string, getType string, designId string, limit int32) ([]objects.JobInfo, error)

	UpdateJob(userId string, jobId string) (objects.JobInfo, error)
	DeleteJob(userId string, jobId string) error

	//TODO would like to not expose these methods as they are for internal use.
	UpdateJobDetails(jobId string, updateType string, msg interface{}) error
}
