#!/usr/bin/env bash

clear

sudo mkdir "/var/log/fledge"
sudo chown -R $USER "/var/log/fledge"

cd /fledge
make local
cd -

../bin/fledgelet start --notifyIp "192.168.0.20" --apiIp "192.168.0.20"