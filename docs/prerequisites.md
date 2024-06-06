# Prerequisites

The target runtime environment is Linux. Development has been mainly conducted under macOS environment. This section describes how to set up a development environment in macOS (Intel chip) and Ubuntu.

## SDK (Data Plane)

The target runtime environment is Linux. Development has been mainly conducted under macOS environment.

The following tools and packages are needed as minimum:
- python 3.9+

You could choose between `conda` or `pyenv` to setup the Python environment.

### Conda

* Install [anaconda](https://www.anaconda.com/download/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) in order to create the environment.

```bash
# Run within the cloned flame directory
cd lib/python/flame
conda create -n flame python=3.9
```

### pyenv

#### mac OS

The following shows how to install the Python environment for SDK in macOS environment.
`brew` is a package management tool in macOS. To install `brew`, refer to [here](https://docs.brew.sh/Installation).
Depending on Linux distributions, several package managers such as `apt`, `yum`, etc. can be used.

Install pyenv (Note : For configuring the pyenv please follow the output of the pyenv init command)
```bash
brew install pyenv
pyenv init
pyenv install 3.9.6
pyenv global 3.9.6
pyenv version

eval "$(pyenv init -)"
echo -e '\nif command -v pyenv 1>/dev/null 2>&1; then\n    eval "$(pyenv init -)"\nfi' >> ~/.bash_profile
```

#### Ubuntu 20.04 or 22.04

The following shows how to install the Python environment for SDK in Ubuntu 20.04 or 22.04.

First, keep package list and their dependencies up to date.
```bash
sudo apt update
```

Install pyenv with the following commands:
```bash
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
    libffi-dev liblzma-dev

curl https://pyenv.run | bash

echo "" >> $HOME/.bashrc
echo "PATH=\"\$HOME/.pyenv/bin:\$PATH\"" >> $HOME/.bashrc
echo "eval \"\$(pyenv init --path)\"" >> $HOME/.bashrc
echo "eval \"\$(pyenv virtualenv-init -)\"" >> $HOME/.bashrc
source $HOME/.bashrc
```

Using `pyenv`, install python version 3.9.6.
```bash
pyenv install 3.9.6
pyenv global 3.9.6
```
To check the version, run `pyenv version` and `python --version`, an example output looks like the following:
```bash
vagrant@flame:~$ pyenv version
3.9.6 (set by /home/vagrant/.pyenv/version)
vagrant@flame:~$ python --version
Python 3.9.6
```

## System (Control Plane)

**Note:** Control plane is only for automated FL job execution. To run examples in SDK, the control plane is not necessary. Therefore, the following configurations can be skipped if SDK is used directly.

The target runtime environment is Linux. Development has been mainly conducted under macOS environment. This section describes how to set up a development environment in macOS (Intel chip) and Ubuntu.

The following tools and packages are needed as minimum:
- go 1.22+
- golangci-lint

After installing above packages, you could try a development setup called `fiab`, an acronym for flame-in-a-box, which is found [here](system/fiab.md).

### macOS (Intel chip required)

The following shows how to install the above packages in macOS environment.
`brew` is a package management tool in macOS. To install `brew`, refer to [here](https://docs.brew.sh/Installation).
Depending on Linux distributions, several package managers such as `apt`, `yum`, etc. can be used.

```bash
brew install go
brew install golangci-lint
```
### Ubuntu 20.04 or 22.04

The following shows how to install the above packages in Ubuntu 20.04 or 22.04.

First, keep package list and their dependencies up to date.
```bash
sudo apt update
```

Install golang and and golangci-lint.
```bash
golang_file=go1.22.3.linux-amd64.tar.gz
curl -LO https://go.dev/dl/$golang_file && tar -C $HOME -xzf $golang_file
echo "PATH=\"\$HOME/go/bin:\$PATH\"" >> $HOME/.bashrc
source $HOME/.bashrc

curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $(go env GOPATH)/bin v1.49.0
golangci-lint --version
```
