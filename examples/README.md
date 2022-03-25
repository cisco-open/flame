# Basic guidelines

This directory contains a few sample examples to explore functionality and workflow of flame.

The examples here can be executed within [fiab](../fiab/README.md) environment.

## Setup
`config.yaml` is a configuration file for local fiab environment. Place it in `$HOME/.flame` folder.

## CLI tool: flamectl
To interact with the flame system, there is a command line (CLI) tool called `flamectl`.

We assume that `flamectl` is in a path specified in PATH environment variable.
By running `make install`, the tool is saved in `$HOME/.flame/bin`.

The following shows several key commands of `flamectl`.

### Basic commands
To see the help menu, run `flamectl` or `flamectl -h`.

To create a design, 
```
flamectl create design <designName>
```

To create a schema, 
```
flamectl create schema <schemaFile> --design <designName>
```

To create a code,
```
flamectl create schema <zippedCodeFile> --design <designName>
```

To create a job,
```
flamectl create job <jobConfigFile>
```

To list a summary of jobs,
```
flamectl get jobs
```

To run a job,
```
flamectl start job <jobId>
```

## Examples

For details on MNIST, visit [here](mnist/README.md).
