#!/usr/bin/env bash
# Copyright 2022 Cisco Systems, Inc. and its affiliates
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


RELEASE_NAME=flame

FILE_OF_INTEREST=helm-chart/deployer/values.yaml

SED_MAC_FIX=
if [[ "$OSTYPE" == "darwin"* ]]; then
    SED_MAC_FIX=\'\'
fi

function pre_start_stop_check {
    # verify that flame.sh script's pods are not running. If they are, throw an error.
    curr_flame_pods=`kubectl get pods -o=name -n flame| grep flame-controller | sed "s/^.\{4\}//"`
    echo "checking for flame.sh pods..."
    if [ -z "$curr_flame_pods" ]; then
        echo "flame.sh script was not run, proceeding with deployer.sh"
        return 1
    else
        echo "flame.sh script was run, cannot proceed with deployer.sh. Stop flame.sh first"
        return 0
    fi
}

function start {
    # install the flame helm chart
    helm install --create-namespace --namespace $RELEASE_NAME $RELEASE_NAME helm-chart/deployer/

    echo "done"
}

function stop {
    helm uninstall --namespace $RELEASE_NAME $RELEASE_NAME

    echo -n "checking if all pods are terminated"

    terminated=false
    while [ "$terminated" == "false" ]; do
        sleep 5
        echo -n "."
        output=`kubectl get pods -n flame -o jsonpath='{.items[*].status.phase}'`
        if [ -z "$output" ]; then
            terminated=true
        fi
    done

    echo "done"

    kubectl delete namespace $RELEASE_NAME
}

function main {
    if [ "$1" == "start" ]; then
        pre_start_stop_check
        local start_stop_check_res=$?
        if [ $start_stop_check_res == 1 ]; then
            start
        else
            echo "Exiting!"
        fi
    elif [ "$1" == "stop" ]; then
        pre_start_stop_check
        local start_stop_check_res=$?
        if [ $start_stop_check_res == 1 ]; then
            stop
        else
            echo "Exiting!"
        fi
    else
        echo "usage: ./deployer.sh <start | stop>"
    fi
}

main "$@"
