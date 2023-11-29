/**
 * Copyright 2023 Cisco Systems, Inc. and its affiliates
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
            "name": "Benchmark of FedOPT Aggregators/Optimizers using MedMNIST example schema v1.0.0 via PyTorch",
            "description": "A simple example of MedMNIST using PyTorch to test out different aggregator algorithms.",
            "roles": [
                {
                    "name": "trainer",
                    "description": "It consumes the data and trains local model",
                    "isDataConsumer": true,
                    "groupAssociation": [
                        {
                            "param-channel": "us"
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
                            "us"
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
