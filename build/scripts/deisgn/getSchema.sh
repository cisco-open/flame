#!/usr/bin/env bash

clear
#../bin/fledgectl design schema get --user "tmpUser" --designId $1

../bin/fledgectl design schema get --user "tmpUser" --designId $1 --type $2 --schemaId $3