#! /usr/bin/env bash

function setup() {
    sudo apt update
    sudo apt upgrade -y

    ERLANG_GPG_KEY=https://packages.erlang-solutions.com/ubuntu/erlang_solutions.asc
    wget -O- $ERLANG_GPG_KEY | sudo apt-key add -
    echo "deb https://packages.erlang-solutions.com/ubuntu focal contrib" | \
	sudo tee /etc/apt/sources.list.d/rabbitmq.list

    sudo apt update

    sudo apt install -y build-essential erlang libsnappy-dev libssl-dev net-tools
}

function install_vernemq() {
    cd /
    VERNEMQ=vernemq
    VERNEMQ_VERSION=1.11.0
    git clone https://github.com/$VERNEMQ/$VERNEMQ.git

    cd $VERNEMQ
    git checkout tags/$VERNEMQ_VERSION
    make rel
}

function main() {
    # no need to call setup function if myungjin-lee/fledge box is in use
    # setup

    install_vernemq
}

main
