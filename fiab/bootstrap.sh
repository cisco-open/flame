#! /usr/bin/env bash

function common {
    # pip3 install monai scikit-learn tqdm

    pip3 install paho-mqtt cloudpickle
}

function controller {
    sudo systemctl enable vernemq
    sudo systemctl start vernemq
}

function worker {
    # do something
    cat /dev/null
}

function main {
    common

    if [ `hostname` == "controller" ]; then
	controller
    else
	worker
    fi
}

main
