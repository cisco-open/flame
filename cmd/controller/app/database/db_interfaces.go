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
	CreateDataset(string, openapi.DatasetInfo) (string, error)
	GetDatasets(string, int32) ([]openapi.DatasetInfo, error)
	GetDatasetById(string) (openapi.DatasetInfo, error)
}

// DesignService is an interface that defines a collection of APIs related to design
type DesignService interface {
	CreateDesign(userId string, info openapi.Design) error
	GetDesign(userId string, designId string) (openapi.Design, error)
	DeleteDesign(userId string, designId string) error
	GetDesigns(userId string, limit int32) ([]openapi.DesignInfo, error)

	CreateDesignSchema(userId string, designId string, info openapi.DesignSchema) error
	GetDesignSchema(userId string, designId string, version string) (openapi.DesignSchema, error)
	GetDesignSchemas(userId string, designId string) ([]openapi.DesignSchema, error)
	UpdateDesignSchema(userId string, designId string, version string, info openapi.DesignSchema) error
	DeleteDesignSchema(userId string, designId string, version string) error

	CreateDesignCode(userId string, designId string, fileName string, fileVer string, fileData *os.File) error
	GetDesignCode(userId string, designId string, version string) ([]byte, error)
	DeleteDesignCode(userId string, designId string, version string) error
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
	GetTaskInfo(string, string, string) (openapi.TaskInfo, error)
	GetTasksInfo(string, string, int32, bool) ([]openapi.TaskInfo, error)
	GetTasksInfoGeneric(string, string, int32, bool, bool) ([]openapi.TaskInfo, error)
}

// TaskService is an interface that defines a collection of APIs related to task
type TaskService interface {
	CreateTasks([]objects.Task, bool) error
	DeleteTasks(string, bool) error
	GetTask(string, string, string) (map[string][]byte, error)
	IsOneTaskInState(string, openapi.JobState) bool
	IsOneTaskInStateWithRole(string, openapi.JobState, string) bool
	MonitorTasks(string) (chan openapi.TaskInfo, chan error, context.CancelFunc, error)
	SetTaskDirtyFlag(string, bool) error
	UpdateTaskStateByFilter(string, openapi.JobState, map[string]interface{}) error
	UpdateTaskStatus(string, string, openapi.TaskStatus) error
}

// ComputeService is an interface that defines a collection of APIs related to computes
type ComputeService interface {
	RegisterCompute(openapi.ComputeSpec) (openapi.ComputeStatus, error)
	GetComputeIdsByRegion(string) ([]string, error)
	GetComputeById(string) (openapi.ComputeSpec, error)
	// UpdateDeploymentStatus call replaces existing agent statuses with received statuses in collection.
	UpdateDeploymentStatus(computeId string, jobId string, agentStatuses map[string]openapi.AgentState) error
}
