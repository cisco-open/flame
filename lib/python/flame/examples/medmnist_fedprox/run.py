# Copyright 2023 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import json

# get trial
if len(sys.argv) != 2:
    # logger?
    raise Exception('Wrong number of arguments; expected 1')

mu = float(sys.argv[1])

# modify json files (jobid, mu-value)
job_ids = ["313a358619cd59012eabeefb", "332a358619cd59012eabeefb", "333a358619cd59012eabeefb", "3333a58619cd59012eabeefb", "3333358619cd59012eabeefb"]
accepted_mu = [1, 0.1, 0.01, 0.001, 0]

task_ids = ["49d06b7526964db86cf37c70e8e0cdb6bd7aa742", "60d06b3526924db76cf37c70e8abcdb6bd7aa74"]
def update_params(filename, mu, num):
    if mu not in [1, 0.1, 0.01, 0.001, 0]:
        raise Exception('mu-value not in {accepted_mu}')
    
    index = accepted_mu.index(mu)
    
    # load json data
    input_filename = "config/train_template.json" if num else "config/agg_template.json"
    file = open(input_filename, 'r')
    data = json.load(file)
    data['optimizer']['kwargs']['mu'] = mu
    data['job']['id'] = job_ids[index]
    base = "https://github.com/GustavBaumgart/flame-datasets/raw/main/medmnist/"
    data['dataset'] = f'{base}site{num}.npz' if num else f'{base}all_val.npz'
    data['taskid'] = f'{task_ids[1]}{num-1}' if num else task_ids[0]
    
    # save json data
    file = open(filename, 'w+')
    file.write(json.dumps(data, indent=4))
    file.close()

num_trainers = 10
for i in range(1, num_trainers+1):
    filename = f'config/trainer{i}.json'
    update_params(filename, mu, i)

filename = 'config/aggregator.json'
update_params(filename, mu, 0)

# run bash file
os.system('bash fedprox.sh')
