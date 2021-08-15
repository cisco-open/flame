#!/usr/bin/env bash

clear
../../bin/fledgectl job changeSchema --user "tmpUser" --jobId $1 --designId $2 --schemaId $3