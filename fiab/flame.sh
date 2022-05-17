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

    echo "mongodb pods are ready; working on enabling roadbalancer for mongodb..."
    # To enable external access for mongodb,
    # uncomment the lines of interest and upgrade the release
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
	sed -i -e "$LINES_OF_INTEREST"" s/^  # /  /" $FILE_OF_INTEREST
    elif [[ "$OSTYPE" == "darwin"* ]]; then
	sed -i '' -e "$LINES_OF_INTEREST"" s/^  # /  /" $FILE_OF_INTEREST
    fi

    helm upgrade --namespace $RELEASE_NAME $RELEASE_NAME helm-chart/
    
    # comment the lines of interest
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
	sed -i -e "$LINES_OF_INTEREST"" s/^  /  # /" $FILE_OF_INTEREST
    elif [[ "$OSTYPE" == "darwin"* ]]; then
	sed -i '' -e "$LINES_OF_INTEREST"" s/^  /  # /" $FILE_OF_INTEREST
    fi

    echo "done"
}

function post_start_config {
    minikube_ip=$(minikube ip)

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
	# Linux OS with resolvconf is assumed
	# not completely tested
	resolver_file=/etc/resolvconf/resolv.conf.d/base
	echo "search flame.test" >> $resolver_file
	echo "nameserver $minikube_ip" >> $resolver_file
	echo "timeout 5" >> $resolver_file

	sudo resolvconf -u
	systemctl disable --now resolvconf.service
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
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
	sed -i '/^$/d' $tmp_file
    elif [[ "$OSTYPE" == "darwin"* ]]; then
	sed -i '' '/^$/d' $tmp_file
    fi

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

    rm -f $tmp_file
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
	# Linux OS with resolvconf is assumed
	# not completely tested
	resolver_file=/etc/resolvconf/resolv.conf.d/base
	sudo sed -i '/search flame.test/d' $resolver_file
	sudo sed -i '/nameserver $minikube_ip/d' $resolver_file
	sudo sed -i '/timeout 5/d' $resolver_file

	sudo resolvconf -u
	systemctl enable --now resolvconf.service
    elif [[ "$OSTYPE" == "darwin"* ]]; then
	resolver_file=/etc/resolver/flame-test
	sudo rm -f $resolver_file
    fi


    # remove dns entry for flame.test domain in coredns
    tmp_file=tmp.bak
    # step 1: save the current entry
    kubectl get configmap coredns -n kube-system -o json | jq -r '.data."Corefile"' > $tmp_file

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
	sed -i '/^$/d' $tmp_file
	# remove last five lines
	for i in {1..5}; do
	    sed -i '$d' $tmp_file
	done
    elif [[ "$OSTYPE" == "darwin"* ]]; then
	sed -i '' '/^$/d' $tmp_file
	# remove last five lines
	for i in {1..5}; do
	    sed -i '' '$d' $tmp_file
	done
    fi

    # step 4: create patch file
    echo "{\"data\": {\"Corefile\": $(jq -R -s < $tmp_file)}}" > $tmp_file

    # step 5: patch configmap of coredns with the updated dns entries
    kubectl patch configmap coredns \
	    -n kube-system \
	    --type merge \
	    -p "$(cat $tmp_file)"

    rm -f $tmp_file
}

function main {
    if [ "$1" == "start" ]; then
	init
	start
	post_start_config
    elif [ "$1" == "stop" ]; then
	stop
	post_stop_cleanup
    else
	echo "usage: ./flame.sh [start|stop]"
    fi
}

main $1
