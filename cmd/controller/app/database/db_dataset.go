package database

import "wwwin-github.cisco.com/eti/fledge/pkg/openapi"

func CreateDataset(userId string, info openapi.DatasetInfo) error {
	return DB.CreateDataset(userId, info)
}
