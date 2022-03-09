FROM golang:1.16.4-buster AS golang-build
WORKDIR /src
ENV CGO_ENABLED=0
RUN --mount=target=. \
    --mount=type=cache,target=/root/.cache/go-build \
    go test -v ./...

RUN --mount=target=. \
    --mount=type=cache,target=/root/.cache/go-build \
    go build -o /fledge/ ./...

RUN mkdir -p /tmp/job-template
COPY templates/job/ /tmp/job-template/

FROM python:3.9.7-buster AS python-build
WORKDIR /src
COPY ./lib/python /src
RUN python3 setup.py bdist_wheel && cp dist/fledge-*.whl /tmp/

FROM myungjinlee/mlpack:cpu
COPY --from=golang-build /fledge/* /usr/bin/

RUN mkdir -p /fledge/deployment
COPY --from=golang-build /tmp/job-template/ /fledge/deployment/

# Install fledge python library
COPY --from=python-build /tmp/*.whl /tmp/
RUN pip3 install /tmp/*.whl && rm -f /tmp/*.whl
# this python script configures python packages
RUN fledge-config
