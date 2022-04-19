## Parallel experiment example

In this example, we will use a MNIST digit recognition dataset and build a **parallel experiment** exmaple.

We will run this example within [fiab](../../docs/03-fiab.md) environment.

### Step 1: create a design
```bash
flamectl create design parallel_exp -d "parallel exp"
```

### Step 2: create a schema for design parallel_exp
```bash
flamectl create schema schema.json --design parallel_exp
```

### Step 3: create (i.e., add) mnist code to the design

```bash
flamectl create code mnist.zip --design parallel_exp
```
Note: to understand relationship between schema and code, unzip adult.zip and check the folder structure in it.

### Step 4: create parallel datasets
```bash
flamectl create dataset dataset_asia_china.json
```
If successful, this command returns the id of the dataset.
```bash
$ flamectl create dataset dataset_asia_china.json
New dataset created successfully
	dataset ID: "62618818edcb3220775de1a4"
$
```
repeat the same step for other two datasets and then copy all three dataset IDs into `job.json`
```json
	"fromSystem": [
	    "62618818edcb3220775de1a4",
        "62618821edcb3220775de1a5",
        "62618829edcb3220775de1a6"
	]
```

### Step 5: create a job
```bash
flamectl create job job.json
```
If successful, this command returns the id of the created job.
For example,
```bash
$ flamectl create job job.json
New job created successfully
	ID: 624888fda8001d773e34de43
	state: ready
$
```

The ids of jobs can be obtained via the following command.
```bash
flamectl get jobs
```
For example,
```bash
$ flamectl get jobs
+--------------------------+-------+--------------------------------+-------------------------------+-------------------------------+
|          JOB ID          | STATE |           CREATED AT           |          STARTED AT           |           ENDED AT            |
+--------------------------+-------+--------------------------------+-------------------------------+-------------------------------+
| 624888fda8001d773e34de43 | ready | 2022-04-02 17:33:49.915 +0000  | 0001-01-01 00:00:00 +0000 UTC | 0001-01-01 00:00:00 +0000 UTC |
|                          |       | UTC                            |                               |                               |
+--------------------------+-------+--------------------------------+-------------------------------+-------------------------------+
$
```

### Step 6: start a job

Assuming the id is `624888fda8001d773e34de43`, run the following command to schedule a job.
```bash
flamectl start job 624888fda8001d773e34de43
```

### Step 7: check progress

```bash
flamectl get tasks 624888fda8001d773e34de43