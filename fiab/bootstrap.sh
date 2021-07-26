#! /usr/bin/env bash

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
    pip3 install paho-mqtt cloudpickle

    # Printing software versions for verification
    echo -e "\n\n"
    echo "Software version"
    echo "- - - - - - - - - "
    go version
    python3 --version
}

function main {
    setup

    if [ `hostname` == "controller" ]; then
	controller
    else
	worker
    fi
}

main
