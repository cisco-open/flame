#! /usr/bin/env bash

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
