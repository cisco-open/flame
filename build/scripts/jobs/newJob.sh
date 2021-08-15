#!/usr/bin/env bash

clear
../../bin/fledgectl job submit --user "tmpUser" --name "sample job" --desc "sample job description" --designId $1 --priority "low" --schemaId $2