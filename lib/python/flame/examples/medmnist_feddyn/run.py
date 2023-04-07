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
import random

# check arguments
if len(sys.argv) != 1:
    raise Exception('Incorrect number of arguments; expected 0')

def randhex(length=24):
    return ''.join([str(random.choice('0123456789abcdef')) for _ in range(length)])

def new_randhex(used=set(), length=24):
    new_id = randhex(length)
    while new_id in used:
        new_id = randhex(length)
    
    used.add(new_id)
    return new_id

used_job_ids = set()
used_task_ids = set()

job = dict()
job['id'] = new_randhex(used=used_job_ids, length=24)
job['name'] = 'medmnist'


def update_params(input_filename, output_filename, **kwargs):
    template_file = open(input_filename, 'r')
    template_data = json.load(template_file)
    template_file.close()
    
    for key in kwargs:
        template_data[key] = kwargs[key]
    
    # save json data
    out_file = open(output_filename, 'w+')
    out_file.write(json.dumps(template_data, indent=4))
    out_file.close()

def get_sites(filename):
    file = open(filename, 'r')
    return [line.strip() for line in file if line.strip()]

# make folders
os.system('rm -rf output')
os.system('mkdir output')
os.system('mkdir output/aggregator')
os.system('mkdir output/trainer')
os.system('mkdir output/config')

sites = get_sites('./trainer/sites.txt')

# aggregator config
update_params('./aggregator/template.json', './output/config/aggregator.json', job=job, taskid=new_randhex(used=used_task_ids, length=40))

# trainer config
for index, site in enumerate(sites):
    update_params('./trainer/template.json', f'./output/config/trainer{index+1}.json', job=job, taskid=new_randhex(used=used_task_ids, length=40), dataset=site)

# run examples
os.chdir('output/trainer')
for i in range(1,len(sites)+1):
    os.system(f'mkdir trainer{i}')
    os.chdir(f'trainer{i}')
    os.system(f'python ../../../trainer/main.py ../../config/trainer{i}.json > log{i}.txt 2>&1 &')
    os.chdir('..')

os.chdir('../aggregator')
os.system('python ../../aggregator/main.py ../config/aggregator.json > log.txt 2>&1 &')
