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

function install_ubuntu_prerequisites {
    install_minikube

    install_kubectl

    install_helm

    install_jq

    install_docker
}

function install_minikube {
    echo "Installing minikube"
    if ! command -v minikube &> /dev/null
    then
        curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
        sudo install minikube-linux-amd64 /usr/local/bin/minikube
        echo "minikube installed"
    else
        echo "minikube already installed. Skipping..."
    fi
}

function install_kubectl {
    echo "Installing kubectl"
    if ! command -v kubectl &> /dev/null
    then
        curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
        sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
        echo "kubectl installed"
    else
        echo "kubectl already installed. Skipping..."
    fi
}

function install_helm {
    echo "Installing helm"
    if ! command -v helm &> /dev/null
    then
        echo "Downloading helm installation script"
        curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
        chmod 700 get_helm.sh
        ./get_helm.sh
        echo "helm installed"
        echo "Deleting helm installation script"
        rm get_helm.sh
    else
        echo "helm already installed. Skipping..."
    fi
}

function install_jq {
    echo "Installing jq"
    if ! command -v jq &> /dev/null
    then
        wget -O jq https://github.com/stedolan/jq/releases/download/jq-1.6/jq-linux64
        chmod +x ./jq
        sudo mv jq /usr/bin
        echo "jq installed"
    else
        echo "jq already installed. Skipping..."
    fi
}

function install_docker {
    echo "Installing docker using convenience script"
    if ! command -v docker &> /dev/null
    then
        echo "Downloading docker installation script"
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        echo "Docker installed"
        echo "Deleting Docker installation script"
        rm get-docker.sh
        echo "Changing docker ownership to non-root user"
        sudo groupadd -f docker
        sudo usermod -aG docker $USER
        echo "to activate changes made to group re-login or run command: newgrp docker"
    else
        echo "docker already installed. Skipping...."
    fi
}