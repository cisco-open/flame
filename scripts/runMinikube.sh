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

echo "
########## Attention! ##########
# This is a MacOS only, internal utility!
# This is supposed to be used only for the development of Flame, only on MacOS devices!
# This script will stop and recreate your minikube environment.
# Run this script at your own risk!
################################
"

if [[ $(uname) != "Darwin" ]]; then
    echo "This script is supposed to run only on MacOS devices!"
    exit 1
fi

force=false

for arg in "$@"; do
    case "$arg" in
        "-f")
            force=true
            ;;
        *)
            echo "unknown option: $arg";exit 1
            ;;
    esac
done

if [[ $force == false ]]; then
    read -p "Are you sure you want to continue? y or n: " </dev/tty response

    if [[ $response == 'n' ]]; then
        exit 0
    fi
fi

set -x;

minikube start --driver=hyperkit --cpus=6 --memory=6g --disk-size 100gb --kubernetes-version=v1.23.8 --alsologtostderr -v=7

minikube ssh "echo 'cat << EOF >  /var/lib/boot2docker/bootlocal.sh
echo "DNS=8.8.8.8" >> /etc/systemd/resolved.conf
systemctl restart systemd-resolved
EOF
chmod 755 /var/lib/boot2docker/bootlocal.sh' > commands.sh && chmod +x commands.sh"

minikube ssh "sudo su root < commands.sh"
minikube stop && minikube start
minikube addons enable ingress --alsologtostderr -v=7
minikube addons enable ingress-dns

../fiab/setup-cert-manager.sh

afplay /System/Library/Sounds/Glass.aiff

set +x;
