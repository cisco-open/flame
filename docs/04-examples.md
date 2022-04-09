# Examples

This section currently presents one example: FL training for MNIST. More examples will follow in the future.

## MNIST

Here we go over how to run MNIST example with [fiab](03-fiab.md) environment.
If `flamectl` command is not found, please refer to [fiab](03-fiab.md).

### Step 1: create a design
```bash
flamectl create design mnist -d "mnist example"
```
This creates a unique name for a  particular task. 

### Step 2: create a schema for design mnist
```bash
flamectl create schema schema.json --design mnist
```
This defines the topology (e.g., type of **Roles** and **Channels**) of this FL task.

### Step 3: create (i.e., add) mnist code to the design

```bash
flamectl create code mnist.zip --design mnist
```
Note: to understand relationship between schema and code, unzip `mnist.zip` and check the folder structure in it.
For example, it should be 
```bash
  adding: aggregator/ (stored 0%)
  adding: aggregator/main.py (deflated 60%)
  adding: trainer/ (stored 0%)
  adding: trainer/main.py (deflated 61%)
```
instead of
```bash
  adding: mnist/ (stored 0%)
  adding: mnist/aggregator/ (stored 0%)
  adding: mnist/aggregator/main.py (deflated 60%)
  adding: mnist/trainer/ (stored 0%)
  adding: mnist/trainer/main.py (deflated 61%)
```
And the folder names should be the same as the **Roles** respectively that you configured in `schema.json`. 


### Step 4: create a dataset

Note: This step is independent of other prior steps. Here the only assumption is that the information on the dataset
is registered in the flame system. Hence, as long as this step is executed before step 5, the MNIST job can be executed
successfully.
```bash
flamectl create dataset dataset.json
```
The last command returns the dataset's ID if successful.
If you want to start a two-trainer example, you need to create one more dataset because flame automatically assigns a trainer to a new dataset.
As the dataset ID is a unique key based on both URL in `dataset.json` and user ID in `${Home}/.flame/config.yaml`, you can modify either URL or user id. Or you can simply duplicate the same dataset's ID in `job.json`.

### Step 5: modify a job specification

With your choice of text editor, modify `job.json` to specify correct dataset's ID and save the change.

### Step 6: create a job
```bash
flamectl create job job.json
```
If successful, this command returns the id of the created job.

The ids of jobs can be obtained via the following command.
```bash
flamectl get jobs
```

### Step 7: start a job
Before staring your job, you can always use `flamectl get` to check each step is set up corretly. For more info, check 
```bash
flamectl get --help
```


Assuming the id is `6131576d6667387296a5ada3`, run the following command to schedule a job.
```bash
flamectl start job 6131576d6667387296a5ada3
```

### Step 8: check progress
To check the status of the job, you can find the following command.
```bash
flamectl get tasks 6131576d6667387296a5ada3
```

Currently, the flame doesn't provide any dedicated UI or tool to check to the process of a job.
To check it, log into a pod and check logs in `/var/log/flame` folder.

Run the following command to list pods running in the minikube.
```bash
kubectl get pods -n flame
```
For example, the output is similar to:
```bash
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
```bash
kubectl exec -it -n flame flame-agent-e276cf6311c723e7bf0693553a0d858d2b75a100--1-bjmb2 -- bash
```

The log for the flame agent (`flamelet`) is `flamelet.log` under `/var/log/flame`.
The log for an ML task is similar to `task-61bd2da4dcaed8024865247e.log` under `/var/log/flame`.


As an alternative, one can check the progress at MLflow UI in the fiab setup.
Run the following command:
```bash
kubectl get svc -n flame  | grep mlflow | awk '{print $4}'
```
The above command returns an IP address (say, 10.104.56.68).
Use the IP address and paste "http://10.104.56.68:5000" on a web browser.

## Hierarchical MNIST
Likewise, the hierarchical FL example follows the same fashion. 

Navigate to `./examples/hier_mnist`

### Step 1:
```bash 
flamectl create design hier_mnist -d "hier_mnist example"
```
### Step 2:
```bash
flamectl create schema schema.json --design hier_mnist
```
The schema defines the topology of this FL task. For more info, please refer to [05-flame-basics](05-flame-basics.md).
### Step 3:
```bash
flamectl create code hier_mnist.zip --design hier_mnist
```
The zip file should contain code of every code specified in the schema.

### Step 4:
```bash
flamectl create dataset dataset_eu_germany.json
flamectl create dataset dataset_eu_uk.json
flamectl create dataset dataset_na_canada.json
flamectl create dataset dataset_na_us.json
```
Flame will assign a trainer to each dataset. As each dataset has a `realm` specified, the middle aggreagator will be created based on the corresponding `groupby` tag. In this case, there will be one middle aggregator for Europe (eu) and one for North America (na).

### Step 5: 
Put all four dataset IDs into `job.json`, and change training hyperparameters as you like.
```json
"fromSystem": [
    "62439c3725fe244585396ad7",
    "6243a10c25fe244585396af0",
    "6243a13625fe244585396af2",
    "6243a14525fe244585396af3"
]
```

### Step 6:
```bash
flamectl create job job.json
```

### Step 7:
```bash
flamectl start job ${Job ID}
```
