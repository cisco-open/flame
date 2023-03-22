## MedMNIST

We use the PathMNIST dataset from (MedMNIST)[https://medmnist.com/] to go over an example of using adaptive aggregator on a data heterogeneity setting, implemented in PyTorch. This example is run within [fiab](../../docs/03-fiab.md) environment.

Note: You may want to add `--insecure` to all the `flamectl` command if you plan to run the example on your local machine only.

### Step 1: create a design

```bash
flamectl create design medmnist -d "MedMNIST"
```

### Step 2: create a schema 

```bash
flamectl create schema schema.json --design medmnist
```

The schema defines the topology of this FL job. For more info, please refer to [05-flame-basics](05-flame-basics.md). Here the schema is the most classic federated learning setting with one server and multiple clients.

### Step 3: add code to the design

```bash
flamectl create code medmnist.zip --design medmnist
```

### Step 4: create datasets

We use NVFlare's NonIID dataset generation script to split the PathMNIST dataset of MedMNIST into 10 non-overlapping portions in a Non-IID fashion. And for each individual dataset, we splitted it into training and validation set in a 8:2 ratio. The following is the data distribution of the training set of all clients:
![train_summary](images/train_summary.png)
And the following is the data distribution of the validation set of all clients:
![val_summary](images/val_summary.png)

```bash
$ flamectl create dataset dataset1.json
New dataset created successfully
	dataset ID: "629a405422f4715eabf99c5e"
```

Copy the Dataset ID into `dataSpec.json`, and repeat for other datasets.

```bash
flamectl create dataset dataset2.json
flamectl create dataset dataset3.json
flamectl create dataset dataset4.json
flamectl create dataset dataset5.json
flamectl create dataset dataset6.json
flamectl create dataset dataset7.json
flamectl create dataset dataset8.json
flamectl create dataset dataset9.json
flamectl create dataset dataset10.json
```

### Step 5: create a job

To illustrate the power of using adaptive aggregation algorithm on the server end of Federated Learning (FL), we provided an example of comparing FedYogi, FedAdaGrad and FedAdam with FedAvg on a Non-IID medical imaging dataset. The way to do it is by changing the server optimizer used in `job.json`.

```bash
$ flamectl create job job.json
New job created successfully
        ID: 62a195b122f4715eabf99c7c
        state: ready
```

If the job is successful created, it returns a job ID.

```bash
$ flamectl get tasks 62a195b122f4715eabf99c7c
+--------------------------+------------------------------------------+--------+-----------+--------------------------------+
|          JOB ID          |                 TASK ID                  |  TYPE  |   STATE   |           TIMESTAMP            |
+--------------------------+------------------------------------------+--------+-----------+--------------------------------+
| 62a195b122f4715eabf99c7c | 076fe91a682c51fa69126a5fab4f08bb124f059f | system | completed | 2022-06-09 14:01:31.334 +0000  |
|                          |                                          |        |           | UTC                            |
| 62a195b122f4715eabf99c7c | 0d919b05824dc89401eaba3330d872c716a0f435 | system | completed | 2022-06-09 14:01:25.771 +0000  |
|                          |                                          |        |           | UTC                            |
| 62a195b122f4715eabf99c7c | 1f61d99fec1ea955fd5d8b5012070f9242f88ec6 | system | completed | 2022-06-09 14:01:25.761 +0000  |
|                          |                                          |        |           | UTC                            |
| 62a195b122f4715eabf99c7c | 25685bb03c44f99a22495ebd413e6a1984566e92 | system | completed | 2022-06-09 14:01:25.76 +0000   |
|                          |                                          |        |           | UTC                            |
| 62a195b122f4715eabf99c7c | 312fd8b1c58e629f3bd74065e29e85f39479978b | system | completed | 2022-06-09 14:01:25.771 +0000  |
|                          |                                          |        |           | UTC                            |
| 62a195b122f4715eabf99c7c | 50bf000c41ca164eeb247dc3691b058da16e7ed3 | system | completed | 2022-06-09 14:01:25.771 +0000  |
|                          |                                          |        |           | UTC                            |
| 62a195b122f4715eabf99c7c | 72c6f27eb75b5c1df15b8a463a6139566ed17f52 | system | completed | 2022-06-09 14:01:25.763 +0000  |
|                          |                                          |        |           | UTC                            |
| 62a195b122f4715eabf99c7c | 997f9e5fb207f94b4d0b4e0b6e1018807e2f5dc4 | system | completed | 2022-06-09 14:01:25.76 +0000   |
|                          |                                          |        |           | UTC                            |
| 62a195b122f4715eabf99c7c | cca2ebc2929ed4bde47efd3b0a0c561dc9470325 | system | completed | 2022-06-09 14:01:25.771 +0000  |
|                          |                                          |        |           | UTC                            |
| 62a195b122f4715eabf99c7c | d4dd50c6260903f7c4d575540269243794b7f075 | system | completed | 2022-06-09 14:01:25.761 +0000  |
|                          |                                          |        |           | UTC                            |
| 62a195b122f4715eabf99c7c | e28704d6b7ad86bed6cc934ae0c25897fde2284e | system | completed | 2022-06-09 14:01:25.771 +0000  |
|                          |                                          |        |           | UTC                            |
+--------------------------+------------------------------------------+--------+-----------+--------------------------------+
```

### Step 6: start running

```bash
flamectl start job 62a195b122f4715eabf99c7c
```

During running, you can check the status of job by going to [http://mlflow.flame.test](http://mlflow.flame.test) or running `flamectl get tasks ${JOB_ID}` on the command line.

### Results

Here we select one of the clients to demonstrate the performance of these server optimizers. If not with the federated learning, this client's best reported validation accuracy is 0.8452.

|   |FedAvg|FedAdam|FedAdaGrad|FedYogi|
|---|---|---|---|---|
|Val Acc|0.9041|0.9092|**0.9158**|0.9090|
|Training Round|77|**21**|31|37|

The validation accuracy was calculated by the weighted summation, in terms of dataset size, of the final global model evaluating on the validation set across all 10 clients respectively. And the training round records the number of rounds required for the global model to achieve 90% of the validation accuracy, from which we see that adaptive optimizer on the server end increases the convergence speed of the federated learning training while still preserving the good accuracy. 

