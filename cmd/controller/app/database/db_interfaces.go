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
	"context"
	"os"

	"github.com/cisco-open/flame/cmd/controller/app/objects"
	"github.com/cisco-open/flame/pkg/openapi"
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
	CreateDataset(string, openapi.DatasetInfo) (string, error)
	GetDatasets(string, int32) ([]openapi.DatasetInfo, error)
	GetDatasetById(string) (openapi.DatasetInfo, error)
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
	CreateJob(string, openapi.JobSpec) (openapi.JobStatus, error)
	DeleteJob(string, string) error
	GetJob(string, string) (openapi.JobSpec, error)
	GetJobById(string) (openapi.JobSpec, error)
	GetJobStatus(string, string) (openapi.JobStatus, error)
	GetJobs(string, int32) ([]openapi.JobStatus, error)
	UpdateJob(string, string, openapi.JobSpec) error
	UpdateJobStatus(string, string, openapi.JobStatus) error
	GetTasksInfo(string, string, int32, bool) ([]openapi.TaskInfo, error)
}

// TaskService is an interface that defines a collection of APIs related to task
type TaskService interface {
	CreateTasks([]objects.Task, bool) error
	DeleteTasks(string, bool) error
	GetTask(string, string, string) (map[string][]byte, error)
	UpdateTaskStatus(string, string, openapi.TaskStatus) error
	IsOneTaskInState(string, openapi.JobState) bool
	IsOneTaskInStateWithRole(string, openapi.JobState, string) bool
	SetTaskDirtyFlag(string, bool) error
	MonitorTasks(string) (chan openapi.TaskInfo, chan error, context.CancelFunc, error)
}
