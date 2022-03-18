# Getting Started

## Prerequisites

The target runtime environment is Linux. Development has been mainly conducted under macOS environment. For now, this section describes how to set up a development environment in macOS.

The following tools and packages are needed as minimum:
- go 1.16+
- golangci-lint
- python 3.9+

The following shows how to install the above packages in macOS environments.
`brew` is a package management tool in macOS. To install `brew`, refer to [here](https://docs.brew.sh/Installation)
Depending on Linux distributions, several package managers such as `apt`, `yum`, etc. can be used.

```
brew install go
brew install golangci-lint
pyenv install 3.9.6
pyenv global 3.9.6
pyenv version

eval "$(pyenv init -)"
echo -e '\nif command -v pyenv 1>/dev/null 2>&1; then\n    eval "$(pyenv init -)"\nfi' >> ~/.bash_profile
```

## Development Setup

This project provides a development setup called fiab, an acronym for fledge-in-a-box, which is found [here](03-fiab.md).
