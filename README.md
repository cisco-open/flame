# Fledge

Fledge is a platform that allows developers to compose and deploy machine learning (ML) training workloads easily.
The system is comprised of a service and a python library. The service manages machine learning workloads,
while a python library facilitates composition of ML workloads.


## Getting started
This repo contains a dev/test environment in a single machine on top of minikube.
The detailed instructions are found [here](fiab/README.md).

### Development setup

The target runtime environment is Linux. Development has been mainly conducted under macOS environment.
For now, this section describes how to set up a development environment in macOS.

The tested version for golang is 1.16+ and the tested version for python is 3.9+.

```
brew install go
brew install golangci-lint
git clone git@github.com:cisco/fledge.git
```

### To compile locally

```
cd fledge
make local
```

### To run a linter
```
cd fledge
go get ./...
golangci-lint run -v
```
