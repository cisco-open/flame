## Distributed Training

We use the MNIST dataset to go over a distributed learning example, which sometimes people use as a baseline to compare with federated learning. This example is run within [fiab](../../docs/03-fiab.md) environment.

Note: You may want to add `--insecure` to all the `flamectl` command if you plan to run the example on your local machine only.

### Step 1: create a design

```bash
flamectl create design distributed_training -d "distributed training"
```

### Step 2: create a schema 

```bash
flamectl create schema schema.json --design distributed_training
```

The schema defines the topology of this FL job. For more info, please refer to [05-flame-basics](05-flame-basics.md).

### Step 3: add code to the design

```bash
flamectl create code distributed_training.zip --design distributed_training
```

### Step 4: create datasets

```bash
$ flamectl create dataset dataset_1.json
New dataset created successfully
	dataset ID: "629e38c756ead26aef5ed5fb"
```

Copy the Dataset ID into `job.json`, and repeat for other datasets.

```bash
flamectl create dataset dataset_2.json
flamectl create dataset dataset_3.json
```

Replace the dataset IDs generated with the ones existing in `job.json`.

### Step 5: create a job

```bash
$ flamectl create job job.json
New job created successfully
	ID: 629e3c61c688c199bcf534a1
	state: ready
```

If the job is successful created, it returns a job ID.

```bash
$ flamectl get tasks 629e3c61c688c199bcf534a1
+--------------------------+------------------------------------------+--------+-------+--------------------------------+
|          JOB ID          |                 TASK ID                  |  TYPE  | STATE |           TIMESTAMP            |
+--------------------------+------------------------------------------+--------+-------+--------------------------------+
| 629e3c61c688c199bcf534a1 | 83d72306a0a8314c136623971a3488c953c9275b | system | ready | 2022-06-06 17:41:53.971 +0000  |
|                          |                                          |        |       | UTC                            |
| 629e3c61c688c199bcf534a1 | ce94e3e964098ca50cc6f8d112cfadaa3a217f93 | system | ready | 2022-06-06 17:41:53.977 +0000  |
|                          |                                          |        |       | UTC                            |
| 629e3c61c688c199bcf534a1 | d7d8d5df84f5ea6957cfa4e67c38989c5db914db | system | ready | 2022-06-06 17:41:53.974 +0000  |
|                          |                                          |        |       | UTC                            |
+--------------------------+------------------------------------------+--------+-------+--------------------------------+
```

There should be only 3 tasks which corresponding to 3 trainers for this particular job because there aren't any aggregators. 

### Step 6: start running

```bash
flamectl start job 629e3c61c688c199bcf534a1
```

During running, you can check the status of job by going to [http://mlflow.flame.test](http://mlflow.flame.test) or running `flamectl get tasks ${JOB_ID}` on the command line.