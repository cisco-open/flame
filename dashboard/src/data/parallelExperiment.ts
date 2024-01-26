/**
 * Copyright 2024 Cisco Systems, Inc. and its affiliates
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

export default {
    schemas: [
        {
            "name": "A simple parallel experiment schema v1.0.0",
            "description": "The schema demonstrates a naive case of parallel experiment with three aggregators that are in Asia, Europe (EU) and North America (NA). Implemented with Keras",
            "roles": [
                {
                    "name": "trainer",
                    "description": "It consumes the data and trains local model",
                    "isDataConsumer": true,
                    "groupAssociation": [
                        {
                            "param-channel": "us"
                        },
                        {
                            "param-channel": "eu"
                        },
                        {
                            "param-channel": "asia"
                        }
                    ]
                },
                {
                    "name": "aggregator",
                    "description": "It aggregates the updates from trainers",
                    "replica": 1,
                    "groupAssociation": [
                        {
                            "param-channel": "us"
                        },
                        {
                            "param-channel": "eu"
                        },
                        {
                            "param-channel": "asia"
                        }
                    ]
                }
            ],
            "channels": [
                {
                    "name": "param-channel",
                    "description": "Model update is sent from trainer to aggregator and vice-versa",
                    "pair": [
                        "trainer",
                        "aggregator"
                    ],
                    "groupBy": {
                        "type": "tag",
                        "value": [
                            "us",
                            "eu",
                            "asia"
                        ]
                    },
                    "funcTags": {
                        "trainer": [
                            "fetch",
                            "upload"
                        ],
                        "aggregator": [
                            "distribute",
                            "aggregate"
                        ]
                    }
                }
            ]
        }
    ]
}
