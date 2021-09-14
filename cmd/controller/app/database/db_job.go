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
	"wwwin-github.cisco.com/eti/fledge/pkg/openapi"
)

func CreateJob(userId string, jobSpec openapi.JobSpec) (openapi.JobStatus, error) {
	return DB.CreateJob(userId, jobSpec)
}

func UpdateJobStatus(userId string, jobId string, jobStatus openapi.JobStatus) error {
	return DB.UpdateJobStatus(userId, jobId, jobStatus)
}

func SubmitJob(userId string, info openapi.JobInfo) (string, error) {
	return DB.SubmitJob(userId, info)
}

func GetJob(userId string, jobId string) (openapi.JobInfo, error) {
	return DB.GetJob(userId, jobId)
}

func GetJobs(userId string, getType string, designId string, limit int32) ([]openapi.JobInfo, error) {
	return DB.GetJobs(userId, getType, designId, limit)
}

//func GetJobsDetailsBy(userId string, getType string, in map[string]string) ([]openapi.JobInfo, error) {
//	return DB.GetJobsDetailsBy(userId, getType, in)
//}

func UpdateJob(userId string, jobId string) (openapi.JobInfo, error) {
	return DB.UpdateJob(userId, jobId)
}

func DeleteJob(userId string, jobId string) error {
	return DB.DeleteJob(userId, jobId)
}

func UpdateJobDetails(jobId string, updateType string, msg interface{}) error {
	return DB.UpdateJobDetails(jobId, updateType, msg)
}
