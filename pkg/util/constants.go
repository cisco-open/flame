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

package util

const (
	ProjectName = "fledge"

	// General
	ALL          = "all"
	ID           = "id"
	Design       = "design"
	Agents       = "agents"
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
	DBFieldMongoID  = "_id"
	DBFieldUserId   = "userid"
	DBFieldId       = "id"
	DBFieldDesignId = "designid"
	DBFieldSchemaId = "schemaid"
	DBFieldJobId    = "jobid"
	DBFieldAgentId  = "agentid"
	DBFieldState    = "state"
	DBFieldRole     = "role"
	DBFieldTaskType = "type"
	DBFieldIsPublic = "ispublic"

	// Port numbers
	ApiServerRestApiPort  = 10100 // REST API port
	NotifierGrpcPort      = 10101 // for notification and push
	ControllerRestApiPort = 10102 // Controller REST API port
	AgentGrpcPort         = 10103 // for fledgelet - application

	// Service names
	Agent      = "fledgelet"
	ApiServer  = "apiserver"
	Controller = "controller"
	Notifier   = "notifier"
	CliTool    = "fledgectl"

	// file permission
	FilePerm0644 = 0644
	FilePerm0700 = 0700
	FilePerm0755 = 0755

	NumTokensInRestEndpoint = 3
	NumTokensInEndpoint     = 2

	TaskConfigFile = "config"
	TaskCodeFile   = "code.zip"

	LogDirPath = "/var/log/" + ProjectName
)
