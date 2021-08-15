#!/usr/bin/env bash

clear
../../bin/fledgectl dev nodes --ip "localhost" --userId "tmpUser" --designId $1 --conf $2