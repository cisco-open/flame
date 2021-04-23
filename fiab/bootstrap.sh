#! /usr/bin/env bash

function common {
    PIP3=$HOME/.local/bin/pip3

    $PIP3 install monai scikit-learn tqdm
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
    if [ `hostname` == "controller" ]; then
	controller
    else
	worker
    fi
}

main
