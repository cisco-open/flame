package database

import (
	"wwwin-github.cisco.com/eti/fledge/pkg/openapi"
)

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
