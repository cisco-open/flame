#!/bin/bash

function printArray() {
  arr=("$@")
  for i in "${arr[@]}"
  do
    echo $i
  done
}

podType="agent"
namespace="fledge"

# get list of pods for namespace
podsList=($(kubectl get pods --namespace $namespace --template '{{range .items}}{{.metadata.name}}{{"\n"}}{{end}}'))

# find all the agent pod using regex
pods+=($(printf "%s\n" "${podsList[@]}" | grep $podType))

nFilteredPods=${#pods[@]}
echo "Found $nFilteredPods pod(s) with the type $podType"
if [ "$nFilteredPods" -lt 1 ]; then
  echo "Cannot find pod with a type $podType"
else
  for p in "${pods[@]}"
  do
    echo "Deleting pod: "$p
    cmd="kubectl delete pod $p --namespace $namespace"
    #echo "Commdn: $cmd"
    #eval $(cmd)
    $cmd
  done
fi

