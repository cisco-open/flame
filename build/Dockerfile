FROM containers.cisco.com/eti-sre/sre-golang-docker:latest

ARG DEBIAN_FRONTEND=noninteractive

ARG GOPATH=/go
ARG BINARY_NAME=fledge
ARG SRC_DIR=$GOPATH/src/wwwin-github.cisco.com/eti/$BINARY_NAME
ARG CODE_COVERAGE
ARG COVER_OUT=coverage.out
ARG HTML_OUT=coverage.html
ARG STATIC_ANALYSIS
ARG SA_OUT=staticanalysis.txt
ARG ARTIFACTORY_USER
ARG ARTIFACTORY_PASSWORD
ARG ARTIFACTORY_URL=https://${ARTIFACTORY_USER}:${ARTIFACTORY_PASSWORD}@engci-maven-master.cisco.com/artifactory/api/go/nyota-go

ENV GO111MODULE=on
ENV GOPRIVATE="wwwin-github.cisco.com"
ENV GONOPROXY="github.com,gopkg.in,go.uber.org"
ENV GOPROXY="${ARTIFACTORY_URL}"

COPY . $SRC_DIR

WORKDIR $SRC_DIR

RUN go mod tidy

RUN if [ "${CODE_COVERAGE}" = "cc" ] ; \
    then go test -v ./... -coverprofile=${COVER_OUT} && \
        go tool cover -html=${COVER_OUT} -o ${HTML_OUT} ; \
    else go test -v ./... ; fi

# TODO: work on githooks later
# Run gofmt checks in case the developer missed installing the hook
# RUN sh $SRC_DIR/githooks/gofmt_checks

# Find out what type of system this Docker contains.  Pull down the correct
# static analysis tool.  Unzip it.
RUN if [ "${STATIC_ANALYSIS}" = "sa" ] ; then \
      latest=$(curl -s https://api.github.com/repos/dominikh/go-tools/releases/latest) && \
      tagname=$(echo "$latest" | grep "tag_name" | cut -d':' -f2 | cut -d'"' -f2) && \
      [ "$(uname)" = Darwin ] && system=darwin || system=linux && \
      echo "tagname == $tagname system == $system" && \
      curl -L https://github.com/dominikh/go-tools/releases/download/${tagname}/staticcheck_${system}_amd64.tar.gz > /tmp/staticcheck_${system}_amd64.tar.gz && \
      tar xzf /tmp/staticcheck_${system}_amd64.tar.gz && \
      find . -name "*.go" -exec staticcheck/staticcheck -checks=inherit,-U1000,-U1001 --debug.no-compile-errors '{}' ';' > ${SRC_DIR}/${SA_OUT} && \
      rm -rf /tmp/staticcheck_* ; \
    fi

# Everything looks good so far, build the code
RUN mkdir -p /fledge
RUN go build -o /fledge ./...

# TODO: the following creates the template files in the docker image
#       this needs to be added dynamically (e.g., via api and database)
# copy helm template for running a job
RUN mkdir -p /tmp/fledge-job
COPY deployment/fledge-job/ /tmp/fledge-job/

# Compile fledge python package
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y python3-setuptools python3-pip
RUN pip3 install wheel
RUN cd lib/python && python3 setup.py bdist_wheel && cp dist/fledge-*.whl /tmp/
