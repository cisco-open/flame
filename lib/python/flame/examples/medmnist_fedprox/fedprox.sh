#!/usr/bin/env bash
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

if [[ $# -ne 0 ]]; then
    echo 'Did not expect any arguments' >&2
    exit 1
fi

cd trainer

for i in {1..10}
do
    rm -rf trainer$i
    mkdir trainer$i
    cd trainer$i
    python ../main.py ../../config/trainer$i.json > log$i.txt 2>&1  &
    cd ..
done

cd ../aggregator
rm -f log.txt
python main.py ../config/aggregator.json > log.txt 2>&1 &
