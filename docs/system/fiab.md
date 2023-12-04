# Flame In A Box (fiab)

## Overview

fiab is a development environment for flame.
The flame system consists of four components: apiserver, controller, notifier and flamelet.
It also includes mongodb as backend state store.


The `flame/fiab` folder contains several scripts to configure and set up the fiab environment.
Thus, the working directory for this guideline is `flame/fiab`.

This development environment is mainly tested under Ubuntu.
The fiab env is also tested under Linux distributions such as Amazon Linux 2, Archlinux, etc.

## Fiab installation guideline
Follow one of the links below that matches operating system under consideration:

* [Ubuntu](fiab-ubuntu.md)
* [Amazon Linux2 with GPU](fiab-amzn2-gpu.md)
