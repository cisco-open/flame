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

FILE_OF_INTEREST=helm-chart/values.yaml
LINES_OF_INTEREST="27,34"

SED_MAC_FIX=
if [[ "$OSTYPE" == "darwin"* ]]; then
    SED_MAC_FIX=\'\'
fi

function init {
    cd helm-chart
    helm dependency update
    cd ..
}

function start {
    # install the flame helm chart
    helm install --create-namespace --namespace $RELEASE_NAME $RELEASE_NAME helm-chart/

    # Wait until mongodb is up
    ready=false
    MONGODB_PODS=(flame-mongodb-0 flame-mongodb-1)
    echo -n "checking mongodb readiness"
    while [ "$ready" == "false" ]; do
	sleep 5
	ready=true
	for pod in ${MONGODB_PODS[@]}; do
	    output=`kubectl get pod -n flame $pod \
		    -o jsonpath='{.status.containerStatuses[0].ready}'`
	    if [ "$output" != "true" ]; then
		echo -n "."
		ready=false
		break
	    fi
	done
    done
    echo "done"

    if [ "$1" == "false" ]; then
	return
    fi

    # Warning: The remaining part in this function is not stable.
    # It can sometimes make mongodb broken.
    # Since exposing mongodb externally is for debugging purposes, the below is
    # only executed when --exposedb is set.
    echo "mongodb pods are ready; working on enabling roadbalancer for mongodb..."
    # To enable external access for mongodb,
    # uncomment the lines of interest and upgrade the release
    sed -i $SED_MAC_FIX -e "$LINES_OF_INTEREST"" s/^  # /  /" $FILE_OF_INTEREST
    
    helm upgrade --namespace $RELEASE_NAME $RELEASE_NAME helm-chart/

    # comment the lines of interest
    sed -i $SED_MAC_FIX -e "$LINES_OF_INTEREST"" s/^  /  # /" $FILE_OF_INTEREST

    # in mac os, sed somehow creates a backup file whose name ends
    # with '' (SED_MAX_FIX string). couldn't figure out why.
    # as a workaround, delete the backup file
    if [[ $SED_MAC_FIX ]]; then
	rm -f $FILE_OF_INTEREST$SED_MAC_FIX
    fi

    echo "done"
}

function post_start_config {
    minikube_ip=$(minikube ip)

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
	subnet=$(ip a show | grep br- | grep inet | awk '{print $2}')
	resolver_file=/etc/systemd/network/minikube.network
	echo "[Match]" | sudo tee $resolver_file > /dev/null
	echo "Name=br*" | sudo tee -a $resolver_file > /dev/null
	echo "[Network]" | sudo tee -a $resolver_file > /dev/null
	echo "Address=$subnet" | sudo tee -a $resolver_file > /dev/null
	echo "DNS=$minikube_ip" | sudo tee -a $resolver_file > /dev/null
	echo "Domains=~flame.test" | sudo tee -a $resolver_file > /dev/null
	sudo systemctl restart systemd-networkd
    elif [[ "$OSTYPE" == "darwin"* ]]; then
	resolver_file=/etc/resolver/flame-test
	echo "domain flame.test" | sudo tee $resolver_file > /dev/null
	echo "nameserver $minikube_ip" | sudo tee -a $resolver_file > /dev/null
	echo "search_order 1" | sudo tee -a $resolver_file > /dev/null
	echo "timeout 5" | sudo tee -a $resolver_file > /dev/null
    fi

    # add dns entry for flame.test domain  in coredns
    tmp_file=tmp.bak
    # step 1: save the current entry
    kubectl get configmap coredns -n kube-system -o json | jq -r '.data."Corefile"' > $tmp_file
    # step 2: remove empty lines
    sed -i $SED_MAC_FIX '/^$/d' $tmp_file

    # step 3: append the dns entry for flame.test domain at the end of file
    echo "flame.test:53 {" | tee -a $tmp_file > /dev/null
    echo "    errors" | tee -a $tmp_file > /dev/null
    echo "    cache 30" | tee -a $tmp_file > /dev/null
    echo "    forward . $minikube_ip" | tee -a $tmp_file > /dev/null
    echo "}" | tee -a $tmp_file > /dev/null

    # step 4: create patch file
    echo "{\"data\": {\"Corefile\": $(jq -R -s < $tmp_file)}}" > $tmp_file

    # step 5: patch configmap of coredns with the updated dns entries
    kubectl patch configmap coredns \
	    -n kube-system \
	    --type merge \
	    -p "$(cat $tmp_file)"

    rm -f $tmp_file $tmp_file$SED_MAC_FIX
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

function post_stop_cleanup {
    minikube_ip=$(minikube ip)

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
	resolver_file=/etc/systemd/network/minikube.network
	sudo rm -f $resolver_file
	sudo systemctl restart systemd-networkd
    elif [[ "$OSTYPE" == "darwin"* ]]; then
	resolver_file=/etc/resolver/flame-test
	sudo rm -f $resolver_file
    fi

    # remove dns entry for flame.test domain in coredns
    tmp_file=tmp.bak
    # step 1: save the current entry
    kubectl get configmap coredns -n kube-system -o json | jq -r '.data."Corefile"' > $tmp_file

    sed -i $SED_MAC_FIX '/^$/d' $tmp_file
    # remove last five lines
    for i in {1..5}; do
	sed -i $SED_MAC_FIX '$d' $tmp_file
    done

    # step 4: create patch file
    echo "{\"data\": {\"Corefile\": $(jq -R -s < $tmp_file)}}" > $tmp_file

    # step 5: patch configmap of coredns with the updated dns entries
    kubectl patch configmap coredns \
	    -n kube-system \
	    --type merge \
	    -p "$(cat $tmp_file)"

    rm -f $tmp_file $tmp_file$SED_MAC_FIX
}

function main {
    exposedb=false
    if [ "$2" == "--exposedb" ]; then
	exposedb=true
    fi

    if [ "$1" == "start" ]; then
	init
	start $exposedb
	post_start_config
    elif [ "$1" == "stop" ]; then
	stop
	post_stop_cleanup
    else
	echo "usage: ./flame.sh <start [--exposedb] | stop>"
    fi
}

main "$@"
