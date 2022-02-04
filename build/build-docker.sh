#!/bin/bash -e
# Syntax build-docker.sh [-i|--image imagename]

PROJECT=fledge
DOCKER_IMAGE=${PROJECT}:latest
BASE_DOCKER_IMAGE=${PROJECT}:base
H_OUT=index.html
S_OUT=staticanalysis.txt

get_artifactory_credentials() {
  if [[ "${ARTIFACTORY_USER}" != "" ]] && [[ "${ARTIFACTORY_PASSWORD}" != "" ]]; then
    return
  fi

  if [[ "${NYOTA_CREDENTIALS_FILE}" = "" ]]; then
    NYOTA_CREDENTIALS_FILE=~/.nyota/credentials
    echo "Using DEFAULT Artifactory credentials file: $NYOTA_CREDENTIALS_FILE"
  else
    echo "Using Artifactory credentials file: $NYOTA_CREDENTIALS_FILE"
  fi

  [[ "$(uname)" = Darwin ]] && NYOTA_CREDENTIALS_FILE_MOD=$(stat -f "%p" "${NYOTA_CREDENTIALS_FILE}" | cut -c4-) || NYOTA_CREDENTIALS_FILE_MOD=$(stat -c "%a" "${NYOTA_CREDENTIALS_FILE}")
  if [[ ${NYOTA_CREDENTIALS_FILE_MOD} != "400" ]]; then
    echo "File ${NYOTA_CREDENTIALS_FILE} must have 400 mod permission"
    exit 1
  fi

  if [[ "$NYOTA_CREDENTIALS_SECTION" == "" ]]; then
    NYOTA_CREDENTIALS_SECTION=default
  fi

  while IFS=' = ' read key value; do
    if [[ ${key} == \[*] ]]; then
      section=${key}
    elif [[ ${value} ]] && [[ ${section} == "[${NYOTA_CREDENTIALS_SECTION}]" ]]; then
      if [[ ${key} == 'artifactory_user' ]]; then
        ARTIFACTORY_USER=${value}
      elif [[ ${key} == 'artifactory_password' ]]; then
        ARTIFACTORY_PASSWORD=${value}
      fi
    fi
  done <${NYOTA_CREDENTIALS_FILE}
}

code_coverage() {
    # extract the H_OUT file from the docker image created
    id=$(docker create ${BASE_DOCKER_IMAGE})
    docker cp ${id}:/go/src/wwwin-github.cisco.com/eti/${PROJECT}/${H_OUT} .
    docker rm -v ${id}
    if [[ ! -d "pipeline/lib" ]] ; then
        echo "Your coverage HTML report is in $H_OUT"
    fi
}

static_analysis() {
    # extract the S_OUT file from the docker image created
    id=$(docker create ${BASE_DOCKER_IMAGE})
    docker cp ${id}:/go/src/wwwin-github.cisco.com/eti/${PROJECT}/${S_OUT} .
    docker rm -v ${id}

    if [[ ! -d "pipeline/lib" ]] ; then
        echo "Your static analysis report is in $S_OUT"
    fi
}

while [[ $# -gt 0 ]]
do
    key="${1}"

    case ${key} in
    -i|--image)
        DOCKER_IMAGE="${2}"
        shift;shift
        ;;
    -h|--help)
        less README.md
        exit 0
        ;;
    -c|--code-coverage)
        CODE_COVERAGE=cc
        shift
        ;;
    -s|--static-analysis)
        STATIC_ANALYSIS=sa
        shift
        ;;
    *) # unknown
        echo Unknown Parameter $1
        exit 4
    esac
done

# get_artifactory_credentials
echo BUILDING DOCKER ${BASE_DOCKER_IMAGE}

# export GO111MODULE=on
# export GOPRIVATE="wwwin-github.cisco.com"
# export GONOPROXY="github.com,gopkg.in,go.uber.org"
# export GOPROXY=https://${ARTIFACTORY_USER}:${ARTIFACTORY_PASSWORD}@engci-maven-master.cisco.com/artifactory/api/go/nyota-go

# docker pull containers.cisco.com/eti-sre/sre-golang-docker:latest
docker build --no-cache \
    -t ${BASE_DOCKER_IMAGE} \
    -f build/Dockerfile \
    --build-arg HTML_OUT=${H_OUT} \
    --build-arg CODE_COVERAGE=${CODE_COVERAGE} \
    --build-arg STATIC_ANALYSIS=${STATIC_ANALYSIS} \
    --build-arg SA_OUT=${S_OUT} \
    .

# if [[ "${CODE_COVERAGE}" = "cc" ]] ; then
#     echo "Generating code coverage"
#     code_coverage
# fi

# if [[ "${STATIC_ANALYSIS}" = "sa" ]] ; then
#     echo "Generating static analysis"
#     static_analysis
# fi

echo BUILDING DOCKER ${DOCKER_IMAGE}
docker build --no-cache -t ${DOCKER_IMAGE} -f build/Imagefile .

# Create fledge worker images
WORKER_IMAGE_NAME=flegde-worker
# FRAMEWORKS=(allinone pytorch tensorflow)
# RESOURCES=(cpu gpu)
FRAMEWORKS=(allinone)
RESOURCES=(cpu)
for framework in ${FRAMEWORKS[@]}; do
    for resource in ${RESOURCES[@]}; do
        target=${framework}-${resource}
        docker build \
               -t ${WORKER_IMAGE_NAME}:{target} \
               -f build/WorkerImagefile \
               --build-arg TARGETIMAGE=${target} .
    done
done

# remove base image and its subordinate images
# docker image rmi ${BASE_DOCKER_IMAGE} || true
