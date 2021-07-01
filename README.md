# Fledge

Fledge is a platform that allows developers to compose and deploy machine learning (ML) training workloads easily.
The system is comprised of a service and a python library. The service manages machine learning workloads,
while a python library facilitates composition of ML workloads.


## Development setup

Development has been mainly conducted under macOS environment. For now, this section describes how to set up
a development environment in macOS.

The tested version for golang is 1.16+ and the tested version for python is 3.8+.

```
brew install go
brew install golangci-lint
git clone git@wwwin-github.cisco.com:eti/fledge.git
```

### To compile locally

```
cd fledge
make local
```

### To run a linter
```
cd fledge
./lint.sh
```
