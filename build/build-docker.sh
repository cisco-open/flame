#!/bin/bash -e
# Syntax build-docker.sh [-i|--image imagename]

PROJECT=fledge
DOCKER_IMAGE=${PROJECT}:latest
BASE_DOCKER_IMAGE=${PROJECT}:base
H_OUT=index.html
S_OUT=staticanalysis.txt

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

echo BUILDING DOCKER ${BASE_DOCKER_IMAGE}

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

if [[ "${DOCKER_IMAGE}" == "fledge-worker"* ]]; then
    # Create fledge worker image
    target=${DOCKER_IMAGE}
    target=${target#"fledge-worker-"}
    target=${target%:*}
    docker --no-cache build -t ${DOCKER_IMAGE} \
           -f build/WorkerImagefile \
           --build-arg TARGETIMAGE=${target} .

elif [[ "${DOCKER_IMAGE}" == "fledge:"* ]]; then
    # Create fledge control plane image
    docker build --no-cache -t ${DOCKER_IMAGE} -f build/Imagefile .

else
    echo "Error: Unknown type $DOCKER_IMAGE"
    exit 1
fi
