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
if len(sys.argv) != 3:
    # logger?
    raise Exception(
        "Wrong number of arguments; expected 2\nExample usage: python run.py (num of trainers to select) (num of total trainers)"
    )

aggr_num = int(sys.argv[1])
total_trainer_num = int(sys.argv[2])

# generate json files
job_id = "622a358619ab59012eabeefb"
task_id = 0


def generate_config(filename, num):
    # load json data
    input_filename = "config/train_template.json" if num else "config/agg_template.json"
    file = open(input_filename, "r")
    data = json.load(file)
    if not num:
        data["selector"]["kwargs"]["aggr_num"] = aggr_num
    data["job"]["id"] = job_id

    data["taskid"] = f"{str(task_id+num)}" if num else str(task_id)

    # save json data
    file = open(filename, "w+")
    file.write(json.dumps(data, indent=4))
    file.close()


# make directory for output files
os.system("mkdir output")
os.system("mkdir output/aggregator")
os.system("mkdir output/trainer")

for i in range(1, total_trainer_num + 1):
    filename = f"config/trainer{i}.json"
    generate_config(filename, i)

    os.chdir("trainer")
    os.system(
        f"python main.py ../config/trainer{i}.json > ../output/trainer/trainer{i}.log 2>&1 &"
    )
    os.chdir("..")

filename = "config/aggregator.json"
generate_config(filename, 0)

os.chdir("aggregator")
os.system(
    "python main.py ../config/aggregator.json > ../output/aggregator/aggregator.log 2>&1 &"
)
os.chdir("..")
