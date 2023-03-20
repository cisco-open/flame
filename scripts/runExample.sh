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

set -x;

source $HOME/.bashrc

exampleName=$1
datasetFileName="dataset"

echo "Starting example: ${exampleName}"

if(($#!=1));
then
    echo "Please provide an example name!"
    exit 1
fi

if [[ $exampleName == "hier_mnist" ]];
then
    datasetFileName="dataset_na_us"
fi

if [[ $exampleName == "medmnist" ]];
then
    datasetFileName="dataset1"
fi

if [[ $exampleName == "parallel_experiment" ]];
then
    datasetFileName="dataset_eu_uk"
fi

if [[ $exampleName == "asyncfl_hier_mnist" ]];
then
    datasetFileName="dataset_eu_uk"
fi

if [[ $exampleName == "distributed_training" ]];
then
    datasetFileName="dataset_1"
fi

echo "Dataset: ${datasetFileName}"

pathPrefix="../examples/${exampleName}/"

echo "${pathPrefix}schema.json"

flamectl create design "${exampleName}" -d "${exampleName} example" --insecure
flamectl create schema "${pathPrefix}schema.json" --design "${exampleName}" --insecure
flamectl create code "${pathPrefix}${exampleName}.zip" --design "${exampleName}" --insecure

datasetData=$(flamectl create dataset "${pathPrefix}${datasetFileName}.json" --insecure)
echo "datasetData: ${datasetData}"

datasetID=$( echo "${datasetData}" | grep -o -e "\".*\"" | tr -d "\"")
echo "datasetID: ${datasetID}"

cp "${pathPrefix}dataSpec.json" "${pathPrefix}tmp.json"

echo "$(jq --arg datasetID "$datasetID" '.fromSystem."0" = [ $datasetID ]' "${pathPrefix}tmp.json")" > "${pathPrefix}result.json"

cp "${pathPrefix}result.json" "${pathPrefix}dataSpec.json"
rm "${pathPrefix}tmp.json"
rm "${pathPrefix}result.json"

#echo "$(jq --arg jobID "test" '.fromSystem."0" = [ $jobID ]' "${pathPrefix}tmp.json")" > result.json
jobID=$( flamectl create job "${pathPrefix}job.json" --insecure | grep "ID: " | tr -d "ID: " )
echo "jobID: ${jobID}"

flamectl start job $jobID --insecure
flamectl get jobs --insecure

set +x;