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

 function install_mac_prerequisites {
     install_hyperkit

     install_docker

     install_minikube

     install_kubectl

     install_helm

     install_jq

     install_robo3t
 }

 function install_hyperkit {
     if ! command -v hyperkit &> /dev/null
     then
         echo "Installing hyperkit"
         brew install hyperkit
         echo "hyperkit installed"
     else
         echo "hyperkit already installed. Skipping..."
     fi
 }

 function install_docker {
     if ! command -v docker &> /dev/null
     then
         echo "Installing docker"
         brew install docker
         echo "docker installed"
     else
         echo "docker already installed. Skipping..."
     fi
 }

 function install_minikube {
     if ! command -v minikube &> /dev/null
     then
         echo "Installing minikube"
         brew install minikube
         echo "minikube installed"
     else
         echo "minikube already installed. Skipping..."
     fi
 }

 function install_kubectl {
     if ! command -v kubectl &> /dev/null
     then
         echo "Installing kubectl"
         brew install kubectl
         echo "kubectl installed"
     else
         echo "kubectl already installed. Skipping..."
     fi
 }

 function install_helm {
     if ! command -v helm &> /dev/null
     then
         echo "Installing helm"
         brew install helm
         echo "helm installed"
     else
         echo "helm already installed. Skipping..."
     fi
 }

 function install_jq {
     if ! command -v jq &> /dev/null
     then
         echo "Installing jq"
         brew install jq
         echo "jq installed"
     else
         echo "jq already installed. Skipping..."
     fi
 }

 function install_robo3t {
     echo "Installing robo3t"
     brew install --cask robo-3t
     echo "robo3t installed"
 }