#! /usr/bin/env bash
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


function controller {
    sudo systemctl enable vernemq
    sudo systemctl start vernemq
}

function worker {
    # do something
    cat /dev/null
}

function setup {
    sudo apt-get update
    sudo apt-get upgrade
    sudo apt-get install build-essential -y

    # go installation
    wget https://golang.org/dl/go1.16.6.linux-amd64.tar.gz
    sudo tar -C /usr/local/ -xzf go1.16.6.linux-amd64.tar.gz
    echo 'export PATH=$PATH:/usr/local/go/bin' >> /home/vagrant/.bashrc
    source /home/vagrant/.bashrc

    # python
    sudo apt install software-properties-common -y
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get update -y
    sudo apt-get install python3.9 python3.9-venv python3.9-dev python3.9-gdbm python3-pip -y
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

    # python packages
    pip3 install paho-mqtt cloudpickle tensorflow torch torchvision scikit-learn

    # Printing software versions for verification
    echo -e "\n\n"
    echo "Software version"
    echo "- - - - - - - - - "
    go version
    python3 --version
}

function main {
    # setup

    if [ `hostname` == "controller" ]; then
      controller
    else
      worker
    fi
}

main
