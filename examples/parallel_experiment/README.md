## Parallel experiment example

In this example, we will use a MNIST digit recognition dataset and build a **parallel experiment** exmaple.

We will run this example within [fiab](../../docs/03-fiab.md) environment.

Note: You may want to add `--insecure` to all the `flamectl` command if you plan to run the example on your local machine only.
### Step 1: create a design
```bash
flamectl create design parallel_experiment -d "parallel exp"
```

### Step 2: create a schema
```bash
flamectl create schema schema.json --design parallel_experiment
```

### Step 3: add mnist code to the design

```bash
flamectl create code parallel_experiment.zip --design parallel_experiment
```
Note: to understand relationship between schema and code, unzip `adult.zip` and check the folder structure in it.

### Step 4: create parallel datasets
```bash
flamectl create dataset dataset_asia_china.json
```
If successful, this command returns the id of the dataset.
```bash
$ flamectl create dataset dataset_asia_china.json
New dataset created successfully
	dataset ID: "62618818edcb3220775de1a4"
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
	ID: 629e36c256ead26aef5ed5f9
	state: ready
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
| 629e36c256ead26aef5ed5f9 | ready | 2022-04-02 17:33:49.915 +0000  | 0001-01-01 00:00:00 +0000 UTC | 0001-01-01 00:00:00 +0000 UTC |
|                          |       | UTC                            |                               |                               |
+--------------------------+-------+--------------------------------+-------------------------------+-------------------------------+
$
```

```bash
$ flamectl get tasks 629e36c256ead26aef5ed5f9

+--------------------------+------------------------------------------+--------+-------+--------------------------------+
|          JOB ID          |                 TASK ID                  |  TYPE  | STATE |           TIMESTAMP            |
+--------------------------+------------------------------------------+--------+-------+--------------------------------+
| 629e36c256ead26aef5ed5f9 | 343ef54d8917ff7a1718a91fa21fa869a4d36828 | system | ready | 2022-06-06 17:17:54.053 +0000  |
|                          |                                          |        |       | UTC                            |
| 629e36c256ead26aef5ed5f9 | 562c6445547dcf3226372a1ebabe0c88d00376e4 | system | ready | 2022-06-06 17:17:54.048 +0000  |
|                          |                                          |        |       | UTC                            |
| 629e36c256ead26aef5ed5f9 | 57dcb4c9b0a4258a40a843d6d6bd3fe133381284 | system | ready | 2022-06-06 17:17:54.05 +0000   |
|                          |                                          |        |       | UTC                            |
| 629e36c256ead26aef5ed5f9 | 58f32c1a1fbf1d3d48a682ec33c2a00a7c176f50 | system | ready | 2022-06-06 17:17:54.045 +0000  |
|                          |                                          |        |       | UTC                            |
| 629e36c256ead26aef5ed5f9 | d8649ebe1e9d126802abc15b41bc471883960ffe | system | ready | 2022-06-06 17:17:54.055 +0000  |
|                          |                                          |        |       | UTC                            |
| 629e36c256ead26aef5ed5f9 | dd7ad325f0215dc179c22d5641c43cb6c23db64a | system | ready | 2022-06-06 17:17:54.058 +0000  |
|                          |                                          |        |       | UTC                            |
+--------------------------+------------------------------------------+--------+-------+--------------------------------+
```

It should generate 6 tasks as expected because it contains 3 FL experiments running in parallel and each experiment has 1 trainer and 1 aggregator, making a total of 6 tasks.

### Step 6: start a job

Assuming the id is `629e36c256ead26aef5ed5f9`, run the following command to schedule a job.
```bash
flamectl start job 629e36c256ead26aef5ed5f9
```

### Step 7: check progress

```bash
flamectl get tasks 629e36c256ead26aef5ed5f9
```
or go to [http://mlflow.flame.test](http://mlflow.flame.test)