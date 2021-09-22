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

package mongodb

import (
	"context"
	"encoding/json"
	"fmt"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo/options"
	"go.uber.org/zap"

	"wwwin-github.cisco.com/eti/fledge/cmd/controller/app/objects"
)

// CreatePayloads creates payload records in payload db collection
func (db *MongoService) CreatePayloads(payloads []objects.Payload) error {
	zap.S().Debugf("Calling CreatePayloads")

	success := false

	// rollback closure in case of error
	defer func() {
		if success {
			return
		}
		// TODO: implement this
	}()

	for _, payload := range payloads {
		cfgData, err := json.Marshal(&payload.JobConfig)
		if err != nil {
			return fmt.Errorf("failed to marshal payload: %v", err)
		}

		filter := bson.M{"jobid": payload.JobId, "agentid": payload.AgentId}
		update := bson.M{
			"$set": bson.M{
				"config": cfgData,
				"code":   payload.ZippedCode,
			},
		}

		after := options.After
		upsert := true
		opts := options.FindOneAndUpdateOptions{
			ReturnDocument: &after,
			Upsert:         &upsert,
		}

		var updatedDoc bson.M
		err = db.payloadCollection.FindOneAndUpdate(context.TODO(), filter, update, &opts).Decode(&updatedDoc)
		if err != nil {
			zap.S().Errorf("Failed to insert a payload: %v", err)
			return err
		}
	}

	success = true
	zap.S().Debugf("Created payloads successufully")

	return nil
}
