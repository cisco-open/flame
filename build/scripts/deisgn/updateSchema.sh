#!/usr/bin/env bash

clear
../../bin/fledgectl design schema update --user "tmpUser" --designId $1 --schemaId $2 --conf "./update_sample_schema.yaml"