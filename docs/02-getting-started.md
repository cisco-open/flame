# Getting Started

## Prerequisites

The target runtime environment is Linux. Development has been mainly conducted under macOS environment. This section describes how to set up a development environment in macOS (Intel chip) and Ubuntu.

The following tools and packages are needed as minimum:
- go 1.16+
- golangci-lint
- python 3.9+

The following shows how to install the above packages in macOS environments.
`brew` is a package management tool in macOS. To install `brew`, refer to [here](https://docs.brew.sh/Installation).
Depending on Linux distributions, several package managers such as `apt`, `yum`, etc. can be used.

```bash
brew install go
brew install golangci-lint
```
Then install pyenv (Note : For configuring the pyenv please follow the output of the pyenv init command)
```bash
brew install pyenv
pyenv init
pyenv install 3.9.6
pyenv global 3.9.6
pyenv version

eval "$(pyenv init -)"
echo -e '\nif command -v pyenv 1>/dev/null 2>&1; then\n    eval "$(pyenv init -)"\nfi' >> ~/.bash_profile
```
The following shows how to install the above packages in Ubuntu.
```bash
sudo apt install golang 
sudo snap install golangci-lint
```
Then install pyenv
```bash
sudo apt-get update; sudo apt-get install make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

curl https://pyenv.run | bash

# pyenv
#Add the following entries into your ~/.bashrc or ~/.bash_profile file:
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"
#restart the shell
exec $SHELL
#OR source the shell
source ~/.bashrc  source ~/.bash_profile

pyenv install 3.9.6
pyenv global 3.9.6
pyenv version
eval "$(pyenv init -)"
echo -e '\nif command -v pyenv 1>/dev/null 2>&1; then\n    eval "$(pyenv init -)"\nfi' >> ~/.bash_profile
```

## Development Setup

This project provides a development setup called fiab, an acronym for flame-in-a-box, which is found [here](03-fiab.md).
