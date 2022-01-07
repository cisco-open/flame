#!/usr/bin/env bash

go install github.com/google/addlicense@v1.0.0

LICENSE_FILE=license.tmp

# update year
year=$(date +'%Y')
cat ../LICENSE | sed -e "s/2021/${year}/" > ${LICENSE_FILE}

# add license for go files
find .. -type f -name *.go -exec ${HOME}/go/bin/addlicense -f ${LICENSE_FILE} '{}' +

# add license for python files
find .. -type f -name *.py -exec ${HOME}/go/bin/addlicense -f ${LICENSE_FILE} '{}' +

rm -f ${LICENSE_FILE}
