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

	"wwwin-github.cisco.com/eti/fledge/pkg/openapi"
)

// CreateDesign - Create a new design template entry in the database.
func CreateDesign(userId string, info openapi.Design) error {
	return DB.CreateDesign(userId, info)
}

func GetDesigns(userId string, limit int32) ([]openapi.DesignInfo, error) {
	return DB.GetDesigns(userId, limit)
}

func GetDesign(userId string, designId string) (openapi.Design, error) {
	return DB.GetDesign(userId, designId)
}

func GetDesignSchema(userId string, designId string, version string) (openapi.DesignSchema, error) {
	return DB.GetDesignSchema(userId, designId, version)
}

func GetDesignSchemas(userId string, designId string) ([]openapi.DesignSchema, error) {
	return DB.GetDesignSchemas(userId, designId)
}

func CreateDesignSchema(userId string, designId string, info openapi.DesignSchema) error {
	return DB.CreateDesignSchema(userId, designId, info)
}

func UpdateDesignSchema(userId string, designId string, version string, info openapi.DesignSchema) error {
	return DB.UpdateDesignSchema(userId, designId, version, info)
}

func CreateDesignCode(userId string, designId string, fileName string, fileVer string, fileData *os.File) error {
	return DB.CreateDesignCode(userId, designId, fileName, fileVer, fileData)
}

func GetDesignCode(userId string, designId string, version string) ([]byte, error) {
	return DB.GetDesignCode(userId, designId, version)
}
