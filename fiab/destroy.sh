#! /usr/bin/env bash

echo "Destroying the worker..."
for i in `vagrant status | grep virtualbox | awk '{ print $1 }'` ;
#for i in `vagrant global-status | grep virtualbox | awk '{ print $2 }'` ;
do
  if [[ $i == *"worker"* ]]; then
    echo "Name : " $i ;
    vagrant destroy -f $i
  fi
done
