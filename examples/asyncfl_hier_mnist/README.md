## Asynchronous Hierarchical Federated Learning with MNIST

This example is based on asynchronous federated learning implementation in the Flame's SDK. The implemention is based on research papers from [here](https://arxiv.org/pdf/2106.06639.pdf) and [here](https://arxiv.org/pdf/2111.04877.pdf).

We use the MNIST dataset to walk through an example of asynchronous hierarchical federated learning with Flame.

We assume that a fiab environment is set up properly. To set it up, refer to [this document](../../docs/03-fiab.md).


**Note**: You need to add `--insecure` to all the `flamectl` command when the example is used in a fiab environment.

### Step 1: create a design

```bash
flamectl create design asyncfl_hier_mnist -d "asynchronous hierarchical FL mnist example"
```

### Step 2: create a schema 

```bash
flamectl create schema schema.json --design asyncfl_hier_mnist
```

The schema defines the topology of this FL job. For more info, please refer to [05-flame-basics](05-flame-basics.md).

### Step 3: add code to the design

```bash
flamectl create code asyncfl_hier_mnist.zip --design asyncfl_hier_mnist
```

### Step 4: create datasets

```bash
$ flamectl create dataset dataset_na_us.json
New dataset created successfully
	dataset ID: "629e3095741b82c266a41478"
```

Copy the Dataset ID into `job.json`, and repeat for other datasets.

```bash
flamectl create dataset dataset_na_canada.json
flamectl create dataset dataset_eu_germany.json
flamectl create dataset dataset_eu_uk.json
```

Replace the dataset IDs generated with the ones existing in `job.json`.

Flame will assign a trainer to each dataset. As each dataset has a realm specified, the middle aggreagator will be created based on the corresponding groupBy tag. In this case, there will be one middle aggregator for Europe (eu) and one for North America (na).

### Step 5: create a job

```bash
$ flamectl create job job.json
New job created successfully
	ID: 629e3185741b82c266a4147b
	state: ready
```

If the job is successful created, it returns a job ID.

```bash
$ flamectl get tasks 629e3185741b82c266a4147b
+--------------------------+------------------------------------------+--------+-------+--------------------------------+
|          JOB ID          |                 TASK ID                  |  TYPE  | STATE |           TIMESTAMP            |
+--------------------------+------------------------------------------+--------+-------+--------------------------------+
| 629e3185741b82c266a4147b | 56b04d963015d199988d4f348f73df1630cd4bf4 | system | ready | 2022-06-06 16:55:33.197 +0000  |
|                          |                                          |        |       | UTC                            |
| 629e3185741b82c266a4147b | 63ce1443fbacf064a938ce17ac7e2279547a2a13 | system | ready | 2022-06-06 16:55:33.203 +0000  |
|                          |                                          |        |       | UTC                            |
| 629e3185741b82c266a4147b | 8b24540425e6ea17d6169e473c45faf06cc807e3 | system | ready | 2022-06-06 16:55:33.21 +0000   |
|                          |                                          |        |       | UTC                            |
| 629e3185741b82c266a4147b | 9af2682ba448be82dd16bd60683858c17bd48998 | system | ready | 2022-06-06 16:55:33.206 +0000  |
|                          |                                          |        |       | UTC                            |
| 629e3185741b82c266a4147b | a0222d56b53bddd7996b135c68c5b37834e12d6c | system | ready | 2022-06-06 16:55:33.213 +0000  |
|                          |                                          |        |       | UTC                            |
| 629e3185741b82c266a4147b | a296e108066d23369bbaa593b8280a387a25f0b4 | system | ready | 2022-06-06 16:55:33.215 +0000  |
|                          |                                          |        |       | UTC                            |
| 629e3185741b82c266a4147b | ae150e449f08c138f95f646ab480a8162ed702fb | system | ready | 2022-06-06 16:55:33.218 +0000  |
|                          |                                          |        |       | UTC                            |
+--------------------------+------------------------------------------+--------+-------+--------------------------------+
```
There should be 7 tasks for this particular job according the schema defined. There will be 1 top aggregator, 2 middle aggregator (1 for na, 1 for eu), and 4 trainers (2 for na, 2 for eu).

### Step 6: start running

```bash
flamectl start job 629e3185741b82c266a4147b
```

During running, you can check the status of job by going to [http://mlflow.flame.test](http://mlflow.flame.test) or running `flamectl get tasks ${JOB_ID}` on the command line.

