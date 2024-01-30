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
)

func TestMongoService_DeleteDataset(t *testing.T) {
	mt := mtest.New(t, mtest.NewOptions().ClientType(mtest.Mock))
	defer mt.Close()
	mt.Run("success", func(mt *mtest.T) {
		db := &MongoService{
			datasetCollection: mt.Coll,
		}
		mt.AddMockResponses(bson.D{{"ok", 1}, {"acknowledged", true},
			{"n", 1}})
		err := db.DeleteDataset("userid", "datasetid")
		assert.Nil(t, err)
	})

	mt.Run("no document deleted", func(mt *mtest.T) {
		db := &MongoService{
			datasetCollection: mt.Coll,
		}
		mt.AddMockResponses(bson.D{{"ok", 1}, {"acknowledged", true},
			{"n", 0}})
		err := db.DeleteDataset("userid", "datasetid")
		assert.NotNil(t, err)
	})
}
