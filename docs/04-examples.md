# Examples

This section currently presents one example: FL training for MNIST. More examples will follow in the future.

## MNIST

Here we go over how to run MNIST example with [fiab](03-fiab.md) environment.

### Step 1: create a design
```
flamectl create design mnist -d "mnist example"
```

### Step 2: create a schema for design mnist
```
flamectl create schema schema.json --design mnist
```

### Step 3: create (i.e., add) mnist code to the design

```
flamectl create code mnist.zip --design mnist
```
Note: to understand relationship between schema and code, unzip mnist.zip and check the folder structure in it.


### Step 4: create a dataset

Note: This step is independent of other prior steps. Here the only assumption is that the information on the dataset
is registered in the flame system. Hence, as long as this step is executed before step 5, the MNIST job can be executed
successfully.
```
flamectl create dataset dataset.json
```
The last command returns the dataset's ID if successful.

### Step 5: modify a job specification

With your choice of text editor, modify job.json to specify correct dataset's ID and save the change.

### Step 6: create a job
```
flamectl create job job.json
```
If successful, this command returns the id of the created job.

The ids of jobs can be obtained via the following command.
```
flamectl get jobs
```

### Step 7: start a job

Assuming the id is `6131576d6667387296a5ada3`, run the following command to schedule a job.
```
flamectl start job 6131576d6667387296a5ada3
```

### Step 8: check progress

Currently, the flame doesn't provide any dedicated UI or tool to check to the process of a job.
To check it, log into a pod and check logs in `/var/log/flame` folder.

Run the following command to list pods running in the minikube.
```
kubectl get pods -n flame
```
For example, the output is similar to:
```
NAME                                                             READY   STATUS    RESTARTS   AGE
flame-agent-e276cf6311c723e7bf0693553a0d858d2b75a100--1-bjmb2   1/1     Running   0          69s
flame-agent-e2b3182eb9c2218d820fc9d2e9443e53c2213a72--1-8mqzn   1/1     Running   0          69s
flame-agent-f5a0b353dc3ca60d24174cbbbece3597c3287f3f--1-qlbkv   1/1     Running   0          69s
flame-apiserver-65d8c7fcf4-2jsm6                                1/1     Running   0          164m
flame-controller-f6c99d8d5-b6dt6                                1/1     Running   0          26m
flame-db-869cccd84c-kvnzn                                       1/1     Running   0          164m
flame-notifier-c59bbcf65-qp4lw                                  1/1     Running   0          164m
mlflow-6dd895c889-npbwv                                          1/1     Running   0          164m
postgres-748c47694c-dvzv8                                        1/1     Running   0          164m
```

To log into an agent pod, run the following command.
```
kubectl exec -it -n flame flame-agent-e276cf6311c723e7bf0693553a0d858d2b75a100--1-bjmb2 -- bash
```

The log for the flame agent (`flamelet`) is `flamelet.log` under `/var/log/flame`.
The log for an ML task is similar to `task-61bd2da4dcaed8024865247e.log` under `/var/log/flame`.


As an alternative, one can check the progress at MLflow UI in the fiab setup.
Run the following command:
```
kubectl get svc -n flame  | grep mlflow | awk '{print $4}'
```
The above command returns an IP address (say, 10.104.56.68).
Use the IP address and paste "http://10.104.56.68:5000" on a web browser.
