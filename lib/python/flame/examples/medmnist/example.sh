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

if [[ $# -ne 2 ]]; then
    echo 'Expected two arguments' >&2
    exit 1
fi

optimizer=$1
framework=$2

case $optimizer in
    fedavg|fedadagrad|fedadam|fedyogi)
    ;;
    *)
	echo 'Expected optimizer to be fedavg, fedadagrad, fedadam, or fedyogi'
	exit 1
esac

case $framework in
    pytorch|keras)
    ;;
    *)
	echo 'Expected framework to be pytorch or keras'
	exit 1
esac

cd trainer/$optimizer

for i in {1..3}
do
    rm -rf $optimizer$i
    mkdir $optimizer$i
    cd $optimizer$i
    python ../../$framework/main.py ../$optimizer$i.json > log$i.txt &
    cd ..
done

cd ../../aggregator
rm -f ${optimizer}_log.txt
python $framework/main.py $optimizer.json > ${optimizer}_log.txt &
