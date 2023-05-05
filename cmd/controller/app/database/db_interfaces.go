// Copyright 2022 Cisco Systems, Inc. and its affiliates
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

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
	ComputeService
}

// DatasetService is an interface that defines a collection of APIs related to dataset
type DatasetService interface {
	// CreateDataset creates a new dataset in the db
	CreateDataset(userId string, info openapi.DatasetInfo) (string, error)

	// GetDatasets returns a list of datasets associated with a user
	GetDatasets(userId string, limit int32) ([]openapi.DatasetInfo, error)

	// GetDatasetById returns the details of a particular dataset
	GetDatasetById(string) (openapi.DatasetInfo, error)
}

// DesignService is an interface that defines a collection of APIs related to design
type DesignService interface {
	// CreateDesign adds a design to the db
	CreateDesign(userId string, info openapi.Design) error

	// GetDesign returns a design associated with the given user and design ids
	GetDesign(userId string, designId string) (openapi.Design, error)

	// DeleteDesign deletes the design from the db
	DeleteDesign(userId string, designId string) error

	// GetDesigns returns a list of designs associated with a user
	GetDesigns(userId string, limit int32) ([]openapi.DesignInfo, error)

	// CreateDesignSchema adds a schema for a design to the db
	CreateDesignSchema(userId string, designId string, info openapi.DesignSchema) error

	// GetDesignSchema returns the schema of a design from the db
	GetDesignSchema(userId string, designId string, version string) (openapi.DesignSchema, error)

	// GetDesignSchemas returns all the schemas associated with the given designId
	GetDesignSchemas(userId string, designId string) ([]openapi.DesignSchema, error)

	// UpdateDesignSchema updates a schema for a design in the db
	UpdateDesignSchema(userId string, designId string, version string, info openapi.DesignSchema) error

	// DeleteDesignSchema deletes the schema of a design from the db
	DeleteDesignSchema(userId string, designId string, version string) error

	// CreateDesignCode adds the code of a design to the db
	CreateDesignCode(userId string, designId string, fileName string, fileVer string, fileData *os.File) error

	// GetDesignCode retrieves the code of a design from the db
	GetDesignCode(userId string, designId string, version string) ([]byte, error)

	// DeleteDesignCode deletes the code of a design from the db
	DeleteDesignCode(userId string, designId string, version string) error
}

// JobService is an interface that defines a collection of APIs related to job
type JobService interface {
	// CreateJob creates a new job
	CreateJob(userId string, spec openapi.JobSpec) (openapi.JobStatus, error)

	// DeleteJob deletes a given job
	DeleteJob(userId string, jobId string) error

	// GetJob gets the job associated with the provided jobId
	GetJob(userId string, jobId string) (openapi.JobSpec, error)

	// GetJobById gets the job associated with the provided jobId
	GetJobById(jobId string) (openapi.JobSpec, error)

	// GetJobStatus get the status of a job
	GetJobStatus(userId string, jobId string) (openapi.JobStatus, error)

	// GetJobs returns the list of jobs associated with a user
	GetJobs(userId string, limit int32) ([]openapi.JobStatus, error)

	// GetJobsByCompute returns the list of jobs for a given computeId that have not been finished yet
	GetJobsByCompute(computeId string) ([]openapi.JobStatus, error)

	// UpdateJob updates the job with the given jobId
	UpdateJob(userId string, jobId string, spec openapi.JobSpec) error

	// UpdateJobStatus updates the status of a job given the user Id, job Id and the openapi.JobStatus
	UpdateJobStatus(userId string, jobId string, status openapi.JobStatus) error

	// GetTaskInfo gets the information of a task given the user Id, job Id and task Id
	GetTaskInfo(string, string, string) (openapi.TaskInfo, error)

	// GetTasksInfo gets the information of tasks given the user Id, job Id and a limit
	GetTasksInfo(string, string, int32, bool) ([]openapi.TaskInfo, error)

	// GetTasksInfoGeneric gets the information of tasks given the user Id, job Id, limit and an option to include completed tasks
	GetTasksInfoGeneric(string, string, int32, bool, bool) ([]openapi.TaskInfo, error)
}

// TaskService is an interface that defines a collection of APIs related to task
type TaskService interface {
	// CreateTasks creates tasks given a set of objects.Task and a flag
	CreateTasks([]objects.Task, bool) error

	// DeleteTasks deletes tasks given the job Id and a flag
	DeleteTasks(string, bool) error

	// GetTask gets the task given the user Id, job Id and task Id
	GetTask(string, string, string) (map[string][]byte, error)

	// IsOneTaskInState evaluates if one of the task is in a certain state given the job Id
	IsOneTaskInState(string, openapi.JobState) bool

	// IsOneTaskInStateWithRole evaluates if one of the tasks is in a certain state and with a specific role given the job Id
	IsOneTaskInStateWithRole(string, openapi.JobState, string) bool

	// MonitorTasks monitors the tasks and returns a TaskInfo channel
	MonitorTasks(string) (chan openapi.TaskInfo, chan error, context.CancelFunc, error)

	// SetTaskDirtyFlag sets the dirty flag for tasks given the job Id and a flag
	SetTaskDirtyFlag(jobId string, dirty bool) error

	// UpdateTaskStateByFilter updates the state of the task using a filter
	UpdateTaskStateByFilter(jobId string, newState openapi.JobState, userFilter map[string]interface{}) error

	// UpdateTaskStatus updates the status of a task given the user Id, job Id, and openapi.TaskStatus
	UpdateTaskStatus(jobId string, taskId string, taskStatus openapi.TaskStatus) error
}

// ComputeService is an interface that defines a collection of APIs related to computes
type ComputeService interface {
	// RegisterCompute registers a compute given a openapi.ComputeSpec
	RegisterCompute(openapi.ComputeSpec) (openapi.ComputeStatus, error)

	// GetComputeIdsByRegion gets all the compute Ids associated with a region
	GetComputeIdsByRegion(string) ([]string, error)

	// GetComputeById gets the compute info given the compute Id
	GetComputeById(string) (openapi.ComputeSpec, error)

	// UpdateDeploymentStatus updates the deployment status given the compute Id, job Id and agentStatuses map
	UpdateDeploymentStatus(computeId string, jobId string, agentStatuses map[string]openapi.AgentState) error
}
