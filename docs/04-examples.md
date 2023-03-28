# Examples

This section showcases one example: FL training for MNIST. More examples are found under examples folder below the top level folder (i.e.,`flame`).
You will find the instructions in the README.md file of each example folder.

## MNIST

Here we go over how to run MNIST example with [fiab](03-fiab.md) environment.
The MNIST example is under `flame/examples/mnist`.
We assume that the current work directory is `flame`.
Change the working directory to mnist by executing the following:
```bash
cd examples/mnist
```

### Caution
Note: all the components in the fiab environment uses a selfsigned certificate.
Hence, certificate verification will fail when `flamectl` is executed.

If a `flamectl` command throws an error like the following, `--insecure` flag should be added to skip the verification.
```console
$ flamectl create design mnist
Failed to create a new design - code: -1, error: Post "https://apiserver.flame.test/foo/designs": x509: certificate signed by unknown authority
```
Note that `--insecure` flag should be used with caution and shouldn't be used in production.

If you get an error saying that `flamectl not found`, please refer to [here](03-c-mac.md#Building-flamectl).

### Step 1: create a design
```bash
flamectl create design mnist -d "mnist example"
```
This creates a unique name for a particular job.

### Step 2: create a schema for design mnist
```bash
flamectl create schema schema.json --design mnist
```
This defines the topology (e.g., type of **Roles** and **Channels**) of this FL job.

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
As the dataset ID is a unique key based on both URL in `dataset.json` and user ID in `${HOME}/.flame/config.yaml`, you can modify either URL or user id. Or you can simply duplicate the same dataset's ID in `dataSpec.json`.

### Step 5: modify a job specification

With your choice of text editor, modify `dataSpec.json` to specify correct dataset's ID and save the change.

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
Before staring your job, you can always use `flamectl get` to check each step is set up correctly. For more info, check 
```bash
flamectl get --help
```


Assuming the id is `6131576d6667387296a5ada3`, run the following command to schedule a job.
```bash
flamectl start job 6131576d6667387296a5ada3
```

### Step 8: check progress
To check the status of the job, you can find the following command.
```console
$ flamectl get tasks 6131576d6667387296a5ada3
+--------------------------+------------------------------------------+--------+---------+--------------------------------+
|          JOB ID          |                 TASK ID                  |  TYPE  |  STATE  |           TIMESTAMP            |
+--------------------------+------------------------------------------+--------+---------+--------------------------------+
| 6131576d6667387296a5ada3 | 0257219f78288b6272393f86f2d4985f674af741 | system | running | 2022-05-21 18:16:48.565 +0000  |
|                          |                                          |        |         | UTC                            |
| 6131576d6667387296a5ada3 | 7ffd8a7b9c015d72e08cb3a5c574f7dddd422bde | system | running | 2022-05-21 18:16:42.881 +0000  |
|                          |                                          |        |         | UTC                            |
+--------------------------+------------------------------------------+--------+---------+--------------------------------+
```

More details on a task can be obtained with the following command:
```console
$ flamectl get task 6131576d6667387296a5ada3 0257219f78288b6272393f86f2d4985f674af741
{
    "jobId": "6131576d6667387296a5ada3",
    "taskId": "0257219f78288b6272393f86f2d4985f674af741",
    "role": "aggregator",
    "type": "system",
    "key": "hidden by system",
    "state": "completed",
    "log": "8:16:25.840425: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA\nTo enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.\n2022-05-21 18:16:42.693490: W tensorflow/python/util/util.cc:348] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.\n2022-05-21 18:16:43,359 | builder_impl.py:780 | INFO | MainThread | copy_assets_to_destination_dir | Assets written to: /tmp/tmppmfrysna/model/data/model/assets\nSuccessfully registered model 'mnist-62892c6a582e4d43984c378c'.\n2022/05/21 18:16:48 INFO mlflow.tracking._model_registry.client: Waiting up to 300 seconds for model version to finish creation.                     Model name: mnist-62892c6a582e4d43984c378c, version 1\nCreated version '1' of model 'mnist-62892c6a582e4d43984c378c'.\n",
    "timestamp": "2022-05-21T18:16:48.565Z"
}
```
The second argument is a task id. The log section shows the last 1000 bytes of logs from a task.


Currently, the flame doesn't provide any dedicated UI or tool to check to the process of a job.
To check it, log into a pod and check logs in `/var/log/flame` folder.

Run the following command to list pods running in the minikube.
```bash
kubectl get pods -n flame
```
For example, the output is similar to:
```bash
NAME                                                             READY   STATUS    RESTARTS   AGE
flame-agent-e276cf6311c723e7bf0693553a0d858d2b75a100--1-bjmb2    1/1     Running   0          69s
flame-agent-e2b3182eb9c2218d820fc9d2e9443e53c2213a72--1-8mqzn    1/1     Running   0          69s
flame-agent-f5a0b353dc3ca60d24174cbbbece3597c3287f3f--1-qlbkv    1/1     Running   0          69s
flame-apiserver-65d8c7fcf4-2jsm6                                 1/1     Running   0          164m
flame-controller-f6c99d8d5-b6dt6                                 1/1     Running   0          26m
flame-db-869cccd84c-kvnzn                                        1/1     Running   0          164m
flame-notifier-c59bbcf65-qp4lw                                   1/1     Running   0          164m
mlflow-6dd895c889-npbwv                                          1/1     Running   0          164m
postgres-748c47694c-dvzv8                                        1/1     Running   0          164m
```

To log into an agent pod, run the following command.
```bash
kubectl exec -it -n flame flame-agent-e276cf6311c723e7bf0693553a0d858d2b75a100--1-bjmb2 -- bash
```

The log for the flame agent (`flamelet`) is `flamelet.log` under `/var/log/flame`.
The log for a task is similar to `task-61bd2da4dcaed8024865247e.log` under `/var/log/flame`.


As an alternative, one can check the progress at MLflow UI in the fiab setup.
Open a browser and go to http://mlflow.flame.test.

For other examples, please visit their particular example directories:
- [Medical Image Multi-class Classification with PyTorch](../examples/medmnist/README.md)
- [Binary Income Classifcation with Tabular Dataset](../examples/adult/README.md) 
- [Toy Example of Hierarchical FL](../examples/hier_mnist/README.md)
- [Toy Example of Parallel Experiments](../examples/parallel_experiment/README.md)
- [Toy Example of Distributed Training](../examples/distributed_training/README.md)
