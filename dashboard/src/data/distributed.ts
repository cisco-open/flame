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
            "name": "A simple schema for distributed training with MQTT backend",
            "description": "This implementation is on Keras using MNIST dataset.",
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
                }
            ],
            "channels": [
                {
                    "description": "Model update is sent from a trainer to another trainer",
                    "groupBy": {
                        "type": "tag",
                        "value": [
                            "us"
                        ]
                    },
                    "name": "param-channel",
                    "pair": [
                        "trainer",
                        "trainer"
                    ],
                    "funcTags": {
                        "trainer": [
                            "ring_allreduce"
                        ]
                    }
                }
            ]
        }
    ]
}
