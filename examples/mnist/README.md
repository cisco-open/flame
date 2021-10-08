## MNIST

Here we go over how to run MNIST example with [fiab](../../fiab/README.md) environment.

### Step 1: create a design
```
fledgectl create design mnist -d "mnist example"
```

### Step 2: create a schema for design mnist
```
fledgectl create schema schema.json --design mnist
```

### Step 3: create (i.e., add) mnist code to the design

```
fledgectl create code mnist.zip --design mnist
```
Note: to understand relationship between schema and code, unzip mnist.zip and check the folder structure in it.


### Step 4: create a dataset

Note: This step is independent of other prior steps. Here the only assumption is that the information on the dataset
is registered in the fledge system. Hence, as long as this step is executed before step 5, the MNIST job can be executed
successfully.
```
fledgectl create dataset dataset.json
```
The last command returns the dataset's ID if successful.

### Step 5: modify a job specification

With your choice of text editor, modify job.json to specify correct dataset's ID and save the change.

### Step 6: create a job
```
fledgectl create job job.json
```
If successful, this command returns the id of the created job.


### Step 7: start a job

Assuming the id is `6131576d6667387296a5ada3`, run the following command to schedule a job.
```
fledgectl start job 6131576d6667387296a5ada3
```

### Step 8: check progress

Currently, the fledge doesn't provide any UI or tool to check to the process of a job.
To check it, log into a docker container and check logs in `/var/log/fledge` folder.

### Caveats
In case fledge is running in the fiab environment, the docker compose file (`docker-compose.yaml`)
needs to be updated for `fledgelet{1,2,3}` containers. In particular, update `FLEDGE_AGENT_ID` environment
varaible based on the task. When a job is first executed, it will fail. To obtain the agent id correctly, 
log into controller container and check its logs to identify agent IDs with the following command.
```
$ docker exec -it fledge-controller bash
root@284be0a78b1d:/# cat /var/log/fledge/controller.log | grep "Creating task for agent"
2021/10/08 19:09:51	[DEBUG]	mongodb/task.go:51	Creating task for agent a07f656a6cc4aa5618ff79f3a0af3a6ba46ce957
2021/10/08 19:09:51	[DEBUG]	mongodb/task.go:51	Creating task for agent 792f51acfcba8b3e6a068e3f761b0a931abeabdb
2021/10/08 19:09:51	[DEBUG]	mongodb/task.go:51	Creating task for agent a2f0520404d18030b34a81352b33f2d625e7e1bc
```
Use the IDs (i.e., a07f656a6cc4aa5618ff79f3a0af3a6ba46ce957, 792f51acfcba8b3e6a068e3f761b0a931abeabdb, a2f0520404d18030b34a81352b33f2d625e7e1bc)
returned from the above `grep` command and update `docker-compose.yaml` accordingly.

Note: In a real environment (e.g., kubernetes), this manual step is not necessary.
