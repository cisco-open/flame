# Examples

## Basic commands
To see the help menu, run `fledgectl` or `fledgectl -h`.


To create a design, 
```
fledgectl create design <designName>
```

To create a schema, 
```
fledgectl create schema --design <designName> --path <schemaFile>
```

To create a code,
```
fledgectl create schema --design <designName> --path <zippedCodeFile>
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
fledgectl run job <jobId>
```

## MNIST example

```
cd mnist

fledgectl create design mnist

fledgectl create schema --design mnist --path schema.json

fledgectl create code --design mnist --path mnist.zip

fledgectl create dataset dataset.json
```
The last command returns the dataset's ID if successful.

Modify job.json to specify correct dataset's ID, and run the following to create a job.
```
fledgectl create job job.json
```
If successful, this commands returns the id of the created job.

Assuming the id is `6131576d6667387296a5ada3`, run the following command to schedule a job.
```
fledgectl run job 6131576d6667387296a5ada3
```
