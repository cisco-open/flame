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

package util

const (
	ProjectName = "flame"

	// General
	ALL          = "all"
	ID           = "id"
	Design       = "design"
	Message      = "message"
	Errors       = "errors"
	GenericError = "GenericError"
	InternalUser = "sys"

	// Status
	Status        = "status"
	StatusSuccess = "Success"
	StatusError   = "Error"

	// Database
	MONGODB = "mongodb"
	MySQL   = "mysql"

	// Database Fields
	//TODO append Field to distinguish the fields
	DBFieldMongoID       = "_id"
	DBFieldUserId        = "userid"
	DBFieldId            = "id"
	DBFieldDesignId      = "designid"
	DBFieldCodes         = "codes"
	DBFieldSchemas       = "schemas"
	DBFieldSchemaId      = "schemaid"
	DBFieldJobId         = "jobid"
	DBFieldTaskId        = "taskid"
	DBFieldState         = "state"
	DBFieldRole          = "role"
	DBFieldTaskLog       = "log"
	DBFieldTaskType      = "type"
	DBFieldIsPublic      = "ispublic"
	DBFieldTaskDirty     = "dirty"
	DBFieldTaskKey       = "key"
	DBFieldTimestamp     = "timestamp"
	DBFieldComputeId     = "computeid"
	DBFieldComputeRegion = "region"
	DBFieldURL           = "url"
	DBFieldAgentStatuses = "agentstatuses"

	// Port numbers
	ApiServerRestApiPort  = 10100 // REST API port
	NotifierGrpcPort      = 10101 // for notification and push
	ControllerRestApiPort = 10102 // Controller REST API port
	AgentGrpcPort         = 10103 // for flamelet - application
	MetaServerPort        = 10104 // meta data update and retrieval (Experimental)

	// Service names
	Agent      = "flamelet"
	ApiServer  = "apiserver"
	Controller = "controller"
	Notifier   = "notifier"
	MetaServer = "metaserver"
	Deployer   = "deployer"
	CliTool    = "flamectl"

	// file permission
	FilePerm0644 = 0644
	FilePerm0700 = 0700
	FilePerm0755 = 0755

	NumTokensInRestEndpoint = 3
	NumTokensInEndpoint     = 2

	TaskConfigFile = "config"
	TaskCodeFile   = "code.zip"

	LogDirPath = "/var/log/" + ProjectName

	DefaultRealm = "default"
	RealmSep     = "/"
)
