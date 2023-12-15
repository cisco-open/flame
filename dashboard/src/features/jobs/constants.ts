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

import { getLabel } from "./utils";

export const optimizer = {
  FEDAVG: "fedavg",
  FEDADAGRAD: "fedadagrad",
  FEDADAM: "fedadam",
  FEDYOGI: "fedyogi",
  FEDBUFF: "fedbuff",
  FEDPROX: "fedprox",
  FEDDYN: "feddyn",
}

export const selector = {
  DEFAULT: "default",
  RANDOM: "random",
  FEDBUFF: "fedbuff",
  OORT: "oort",
}

export const SELECTOR_DEFAULT_OPTIONS = {
  [selector.DEFAULT]: [],
  [selector.RANDOM]: [
    { arg: 'k', value: 10 },
  ],
  [selector.FEDBUFF]: [
    { arg: 'c', value: 50 },
  ],
  [selector.OORT]: [
    { arg: 'agg_num', value: 10 },
  ]
}

export const OPTIMIZER_DEFAULT_OPTIONS = {
  [optimizer.FEDADAM]: [
    { arg: 'beta_1', value: 0.9 },
    { arg: 'beta_2', value: 0.99 },
    { arg: 'eta', value: 0.1 },
    { arg: 'tau', value: 0.001 },
  ],
  [optimizer.FEDADAGRAD]: [
    { arg: 'beta_1', value: 0 },
    { arg: 'eta', value: 0.1 },
    { arg: 'tau', value: 0.01 },
  ],
  [optimizer.FEDYOGI]: [
    { arg: 'beta_1', value: 0.9 },
    { arg: 'beta_2', value: 0.99 },
    { arg: 'eta', value: 0.1 },
    { arg: 'tau', value: 0.001 },
  ],
  [optimizer.FEDPROX]: [
    { arg: 'mu', value: 0.01 },
  ],
  [optimizer.FEDDYN]: [
    { arg: 'alpha', value: 0.01 },
  ],
  [optimizer.FEDAVG]: [],
  [optimizer.FEDBUFF]: [],
}

export const BACKEND_OPTIONS = [
  {
    name: 'mqtt',
    id: 1
  },
  {
    name: 'p2p',
    id: 2
  }
];

export const DEFAULT_HYPERPARAMETERS = [
  {
    key: 'rounds',
    value: '1',
    id: 1,
  },
  {
    key: 'epochs',
    value: '1',
    id: 2,
  },
  {
    key: 'batchSize',
    value: 16,
    id: 3,
  },
]

export const SELECTOR_OPTION = {
  random: ['k'],
  oort: ['aggr_num'],
  fedbuff: ['c']
}

export const OPTIMIZER_OPTION = {
  fedadagrad: ['beta_1', 'beta_2', 'eta', 'tau'],
  fedadam: ['beta_1', 'beta_2', 'eta', 'tau'],
  fedavg: [], // https://github.com/cisco-open/flame/blob/main/lib/python/flame/optimizer/fedavg.py
  fedbuff: [], // https://github.com/cisco-open/flame/blob/main/lib/python/flame/optimizer/fedbuff.py
  feddyn: ['alpha'],
  fedprox: ['mu'],
  fedyogi: ['beta_1', 'beta_2', 'eta', 'tau'],
}

export const hyperparameters = {
  ROUNDS: 'rounds',
  EPOCHS: 'epochs',
  BATCH_SIZE: 'batchSize',
  LEARNING_RATE: 'learningRate',
  WEIGHT_DECAY: 'weightDecay',
  AGG_GOAL: 'aggGoal',
  CONCURRENCY: 'concurrency',
  CUSTOM: 'custom',
}

export enum HYPERPARAMETER_TYPE {
  predefined,
  custom
}

export const HYPERPARAMETER_OPTIONS = [
  {
    value: hyperparameters.ROUNDS,
    label: getLabel(hyperparameters.ROUNDS),
    id: 1,
  },
  {
    value: hyperparameters.LEARNING_RATE,
    label: getLabel(hyperparameters.LEARNING_RATE),
    id: 2,
  },
  {
    value: hyperparameters.WEIGHT_DECAY,
    label: getLabel(hyperparameters.WEIGHT_DECAY),
    id: 3,
  },
  {
    value: hyperparameters.BATCH_SIZE,
    label: getLabel(hyperparameters.BATCH_SIZE),
    id: 4,
  },
  {
    value: hyperparameters.EPOCHS,
    label: getLabel(hyperparameters.EPOCHS),
    id: 5,
  },
  {
    value: hyperparameters.AGG_GOAL,
    label: getLabel(hyperparameters.AGG_GOAL),
    id: 6,
  },
  {
    value: hyperparameters.CONCURRENCY,
    label: getLabel(hyperparameters.CONCURRENCY),
    id: 7,
  },
  {
    value: hyperparameters.CUSTOM,
    label: getLabel(hyperparameters.CUSTOM),
    id: 8,
  },
]

export const JOB_PRIORITY_OPTIONS = [
  {
    label: 'Low',
    value: 'low',
    id: 1,
  },
  {
    label: 'Medium',
    value: 'medium',
    id: 2,
  },
  {
    label: 'High',
    value: 'high',
    id: 3,
  }
];

export const SELECTOR_OPTIONS = [
  {
    value: selector.DEFAULT,
    label: getLabel(selector.DEFAULT),
    id: 1,
  },
  {
    value: selector.RANDOM,
    label: getLabel(selector.RANDOM),
    id: 2,
  },
  {
    value: selector.FEDBUFF,
    label: getLabel(selector.FEDBUFF),
    id: 3,
  },
  {
    value: selector.OORT,
    label: getLabel(selector.OORT),
    id: 4,
  }
];

export const OPTIMIZER_OPTIONS = [
  {
    value: optimizer.FEDAVG,
    label: getLabel(optimizer.FEDAVG),
    id: 1,
  },
  {
    value: optimizer.FEDADAGRAD,
    label: getLabel(optimizer.FEDADAGRAD),
    id: 2,
  },
  {
    value: optimizer.FEDADAM,
    label: getLabel(optimizer.FEDADAM),
    id: 3,
  },
  {
    value: optimizer.FEDYOGI,
    label: getLabel(optimizer.FEDYOGI),
    id: 4,
  },
  {
    value: optimizer.FEDBUFF,
    label: getLabel(optimizer.FEDBUFF),
    id: 5,
  },
  {
    value: optimizer.FEDPROX,
    label: getLabel(optimizer.FEDPROX),
    id: 6,
  },
  {
    value: optimizer.FEDDYN,
    label: getLabel(optimizer.FEDDYN),
    id: 7,
  },
];