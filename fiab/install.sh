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


function macos_installation {
    echo "macos: to be implemented"
}

function ubuntu_installation {
    source ./ubuntu.sh
    install_ubuntu_prerequisites
}

function amzn2_installation {
    # create a working directory for amzn2 setting
    mkdir amzn2
    pushd amzn2

    # set up golang compilation env
    wget https://storage.googleapis.com/golang/getgo/installer_linux
    chmod +x ./installer_linux
    ./installer_linux
    source ~/.bash_profile

    # download cri-docker
    git clone https://github.com/Mirantis/cri-dockerd.git
    pushd cri-dockerd
    mkdir bin
    go build -o bin/cri-dockerd

    # install cri-docker
    sudo install -o root -g root -m 0755 bin/cri-dockerd /usr/bin/cri-dockerd
    sudo cp -a packaging/systemd/* /etc/systemd/system
    sudo systemctl daemon-reload
    sudo systemctl enable cri-docker.service
    sudo systemctl enable --now cri-docker.socket
    popd # get out of cri-dockerd

    # install crictl
    VERSION="v1.25.0"
    wget https://github.com/kubernetes-sigs/cri-tools/releases/download/$VERSION/crictl-$VERSION-linux-amd64.tar.gz
    sudo tar zxvf crictl-$VERSION-linux-amd64.tar.gz -C /usr/local/bin
    rm -f crictl-$VERSION-linux-amd64.tar.gz

    # install minikube
    curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-latest.x86_64.rpm
    sudo rpm -Uvh minikube-latest.x86_64.rpm

    # install kubectl
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
    sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

    # install helm
    HELM_VERSION=v3.10.2
    curl -LO https://get.helm.sh/helm-$HELM_VERSION-linux-amd64.tar.gz
    tar -zxvf helm-$HELM_VERSION-linux-amd64.tar.gz
    sudo mv linux-amd64/helm /usr/local/bin/helm

    # clean up the working directory for amzn2
    popd
    rm -rf amzn2
}

function main {
    os=${@:$OPTIND:1}
    shift;

    case $os in
	"macos")
	    echo "Starting installation for Mac OS"
	    macos_installation
	    ;;
	
	"ubuntu")
	    echo "Starting installation for Ubuntu"
	    ubuntu_installation
	    ;;
	"amzn2")
	    echo "Starting installation for Amazon Linux 2"
	    amzn2_installation
	    ;;
	*)
	    echo "usage: ./setup.sh <macos | ubuntu | amzn2>"
	    ;;
    esac
}

main "$@"
