// Copyright 2023 Cisco Systems, Inc. and its affiliates
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

package mongodb

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo/integration/mtest"

	"github.com/cisco-open/flame/pkg/openapi"
)

func TestMongoService_UpdateDeploymentStatus(t *testing.T) {
	mt := mtest.New(t, mtest.NewOptions().ClientType(mtest.Mock))
	defer mt.Close()
	mt.Run("success", func(mt *mtest.T) {
		db := &MongoService{
			deploymentCollection: mt.Coll,
		}
		mt.AddMockResponses(mtest.CreateSuccessResponse(), mtest.CreateCursorResponse(1, "flame.deployment", mtest.FirstBatch, bson.D{}))
		err := db.UpdateDeploymentStatus("test jobid","test compute id",
			map[string]openapi.AgentState{"test task id": openapi.AGENT_DEPLOY_SUCCESS,
		})
		assert.Nil(t, err)
	})
	mt.Run("status update failure", func(mt *mtest.T) {
		db := &MongoService{
			deploymentCollection: mt.Coll,
		}
		mt.AddMockResponses(
			mtest.CreateCursorResponse(1, "flame.deployment", mtest.FirstBatch, bson.D{{"_id", "1"}}))
		err := db.UpdateDeploymentStatus("test jobid", "test compute id",
			map[string]openapi.AgentState{"test task id": openapi.AGENT_DEPLOY_SUCCESS},
		)
		assert.NotNil(t, err)
	})
}
