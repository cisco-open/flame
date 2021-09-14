package database

import (
	"os"

	"wwwin-github.cisco.com/eti/fledge/pkg/openapi"
)

// StoreCollection provides collection of all the db stores in the application.
type StoreCollection interface {
	DatasetStore
	DesignStore
	JobStore
}

type DatasetStore interface {
	CreateDataset(userId string, info openapi.DatasetInfo) error
}

// DesignStore is the collection of db APIs related to the designs
//TODO for all get methods - explicitly specify the fields to be retured as part of the object
type DesignStore interface {
	CreateDesign(userId string, info openapi.Design) error
	GetDesign(userId string, designId string) (openapi.Design, error)
	GetDesigns(userId string, limit int32) ([]openapi.DesignInfo, error)

	CreateDesignSchema(userId string, designId string, info openapi.DesignSchema) error
	GetDesignSchema(userId string, designId string, version string) (openapi.DesignSchema, error)
	GetDesignSchemas(userId string, designId string) ([]openapi.DesignSchema, error)
	UpdateDesignSchema(userId string, designId string, version string, info openapi.DesignSchema) error

	CreateDesignCode(userId string, designId string, fileName string, fileVer string, fileData *os.File) error
	GetDesignCode(userId string, designId string, version string) ([]byte, error)
}

type JobStore interface {
	CreateJob(userId string, jobSpec openapi.JobSpec) (openapi.JobStatus, error)
	UpdateJobStatus(userId string, jobId string, jobStatus openapi.JobStatus) error

	SubmitJob(userId string, info openapi.JobInfo) (string, error)

	GetJob(userId string, jobId string) (openapi.JobInfo, error)
	GetJobs(userId string, getType string, designId string, limit int32) ([]openapi.JobInfo, error)
	//GetJobsDetailsBy(userId string, getType string, in map[string]string) ([]openapi.JobInfo, error)

	UpdateJob(userId string, jobId string) (openapi.JobInfo, error)
	DeleteJob(userId string, jobId string) error

	//TODO would like to not expose these methods as they are for internal use.
	UpdateJobDetails(jobId string, updateType string, msg interface{}) error
}
