# Basic guidelines

This directory contains a few sample examples to explore functionality and workflow of fledge.

The examples here can be executed within [fiab](../fiab/README.md) environment.

## Setup
`config.yaml` is a configuration file for local fiab environment. Place it in `$HOME/.fledge` folder.

## CLI tool: fledgelet
To interact with the fledge system, there is a command line (CLI) tool called `fledgelet`.

Note that `fledgectl` produces logs in `/var/log/fledge/fledgectl.log`.
Therefore, make sure that `fledgectl` is executed as root or modify access permission for `/var/log/fledge` appropriately.
We assume that `fledgectl` is in a path specified in PATH environment variable.
When the binary is created by `make local`, the tool is stored in `build/bin`.

The following shows several key commands of `fledgelet`.

### Basic commands
To see the help menu, run `fledgectl` or `fledgectl -h`.

To create a design, 
```
fledgectl create design <designName>
```

To create a schema, 
```
fledgectl create schema <schemaFile> --design <designName>
```

To create a code,
```
fledgectl create schema <zippedCodeFile> --design <designName>
```

To create a job,
```
fledgectl create job <jobConfigFile>
```

To list a summary of jobs,
```
fledgectl get jobs
```

To run a job,
```
fledgectl start job <jobId>
```

## Examples

For details on MNIST, visit [here](mnist/README.md).
