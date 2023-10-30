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
