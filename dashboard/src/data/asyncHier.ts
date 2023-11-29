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
            "name": "A simple hierarchical FL MNIST example schema",
            "description": "a sample schema to demostrate the hierarchical FL setting",
            "roles": [
                {
                    "name": "trainer",
                    "description": "It consumes the data and trains local model",
                    "isDataConsumer": true,
                    "groupAssociation": [
                        {
                            "param-channel": "eu"
                        },
                        {
                            "param-channel": "na"
                        }
                    ]
                },
                {
                    "name": "middle-aggregator",
                    "description": "It aggregates the updates from trainers",
                    "groupAssociation": [
                        {
                            "param-channel": "eu",
                            "global-channel": "default"
                        },
                        {
                            "param-channel": "na",
                            "global-channel": "default"
                        }
                    ]
                },
                {
                    "name": "top-aggregator",
                    "description": "It aggregates the updates from middle-aggregator",
                    "groupAssociation": [
                        {
                            "global-channel": "default"
                        }
                    ]
                }
            ],
            "channels": [
                {
                    "name": "param-channel",
                    "description": "Model update is sent from trainer to middle-aggregator and vice-versa",
                    "pair": [
                        "trainer",
                        "middle-aggregator"
                    ],
                    "groupBy": {
                        "type": "tag",
                        "value": [
                            "eu",
                            "na"
                        ]
                    },
                    "funcTags": {
                        "trainer": [
                            "fetch",
                            "upload"
                        ],
                        "middle-aggregator": [
                            "distribute",
                            "aggregate"
                        ]
                    }
                },
                {
                    "name": "global-channel",
                    "description": "Model update is sent from middle-aggregator to top-aggregator and vice-versa",
                    "pair": [
                        "top-aggregator",
                        "middle-aggregator"
                    ],
                    "groupBy": {
                        "type": "tag",
                        "value": [
                            "default"
                        ]
                    },
                    "funcTags": {
                        "top-aggregator": [
                            "distribute",
                            "aggregate"
                        ],
                        "middle-aggregator": [
                            "fetch",
                            "upload"
                        ]
                    }
                }
            ]
        }
    ]
}
