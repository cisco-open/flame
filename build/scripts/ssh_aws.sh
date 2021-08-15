#!/bin/bash

function printArray() {
  arr=("$@")
  for i in "${arr[@]}"
  do
    echo $i
  done
}

podType=$1
namespace="fledge"
# get list of pods for namespace
podsList=($(kubectl get pods --namespace $namespace --template '{{range .items}}{{.metadata.name}}{{"\n"}}{{end}}'))

# check if pod type is provided
if [ "$#" -ne 1 ]; then
    echo "Must provide pod type"
    printArray "${podsList[@]}"
    exit
fi

# find the pod using regex
pods+=($(printf "%s\n" "${podsList[@]}" | grep $podType))

nFilteredPods=${#pods[@]}
echo "Found $nFilteredPods pod(s) with the type $podType"
if [ "$nFilteredPods" -lt 1 ]; then
  echo "Cannot find pod with a type $podType"
elif [ "$nFilteredPods" -gt 1  ]; then
    echo -e "Multiple pods with the type $podType found. Be more specific with the podType keyword."
else
  cmd="kubectl exec --namespace $namespace --stdin --tty $pods -- /bin/bash"
  echo "$cmd"
  $cmd
fi

