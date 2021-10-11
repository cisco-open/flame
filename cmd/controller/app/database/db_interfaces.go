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

package database

import (
	"os"

	"github.com/cisco/fledge/cmd/controller/app/objects"
	"github.com/cisco/fledge/pkg/openapi"
)

// DBService defines a total set of collections for the application
type DBService interface {
	DatasetService
	DesignService
	JobService
	TaskService
}

// DatasetService is an interface that defines a collection of APIs related to dataset
type DatasetService interface {
	CreateDataset(userId string, info openapi.DatasetInfo) (string, error)
	GetDatasetById(datasetId string) (openapi.DatasetInfo, error)
}

// DesignService is an interface that defines a collection of APIs related to design
type DesignService interface {
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

// JobService is an interface that defines a collection of APIs related to job
type JobService interface {
	CreateJob(userId string, jobSpec openapi.JobSpec) (openapi.JobStatus, error)
	GetJob(userId string, jobId string) (openapi.JobSpec, error)
	GetJobById(jobId string) (openapi.JobSpec, error)
	GetJobStatus(userId string, jobId string) (openapi.JobStatus, error)
	UpdateJobStatus(userId string, jobId string, jobStatus openapi.JobStatus) error

	DeleteJob(userId string, jobId string) error
}

// TaskService is an interface that defines a collection of APIs related to task
type TaskService interface {
	CreateTasks([]objects.Task) error
	DeleteTasks(string) error
	GetTask(string, string) (map[string][]byte, error)
	UpdateTaskStatus(string, string, openapi.TaskStatus) error
}
