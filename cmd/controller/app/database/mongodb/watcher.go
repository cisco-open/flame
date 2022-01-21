// Copyright (c) 2022 Cisco Systems, Inc. and its affiliates
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
	"strings"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"go.uber.org/zap"
)

const (
	errChannelLen = 2
)

type streamWatcher struct {
	ctx    context.Context
	cancel context.CancelFunc

	coll     *mongo.Collection
	pipeline mongo.Pipeline
	opts     *options.ChangeStreamOptions

	streamCh chan bson.M
	errCh    chan error
}

func (db *MongoService) newstreamWatcher(coll *mongo.Collection, pipeline mongo.Pipeline, opts *options.ChangeStreamOptions,
	chanLen int) *streamWatcher {
	ctx, cancel := context.WithCancel(context.Background())

	return &streamWatcher{
		ctx:    ctx,
		cancel: cancel,

		coll:     coll,
		pipeline: pipeline,
		opts:     opts,

		streamCh: make(chan bson.M, chanLen),
		errCh:    make(chan error, errChannelLen),
	}
}

func (sw *streamWatcher) watch() error {
	changeStream, err := sw.coll.Watch(sw.ctx, sw.pipeline, sw.opts)
	if err != nil {
		return err
	}

	go func() {
		defer changeStream.Close(sw.ctx)

		zap.S().Debug("Started watcher")

		for changeStream.Next(sw.ctx) {
			var event bson.M
			if err := changeStream.Decode(&event); err != nil {
				zap.S().Debugf("Failed to decode event: %v", err)

				sw.errCh <- err
				break
			}

			// push the event to a stream channel
			sw.streamCh <- event
		}

		if err := changeStream.Err(); err != nil {
			// if the error is not because of context cancellation, send the error
			if !strings.Contains(err.Error(), context.Canceled.Error()) {
				sw.errCh <- err
				zap.S().Debugf("Error on change stream: %v", err)
			}
		}

		zap.S().Debug("Finished watcher")
	}()

	return nil
}
