#!/usr/bin/env bash

clear
../../bin/fledgectl dev updateNodes --ip "localhost" --userId "tmpUser" --designId $1 --conf $2