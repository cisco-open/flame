## MNIST in non-orchestration mode

Here we go over how to run MNIST example in non-orchestration mode within [fiab](../../docs/03-fiab.md) environment.
In non-orchestration mode, participants drive participation. Hence, participants should bring their dataset along with compute.

This example is almost identical to the mnist example shown [here](../mnist/README.md) with a few configuration changes.
To see differences, use `diff`. For example, do `diff schema.json ../mnist/schema.json`.

### Prerequisites

We run the MNIST usecase (shown [here](../mnist/README.md)) in a local cluster. We also use a local mqtt broker.
As opposed to the MNIST usecase, in this usecase, we need a container that runs outside the local cluster.
The domain name assigned to the broker is also locally available.
Therefore, a local mqtt broker needs to be replaced with a public mqtt broker to enable the backend communication
across the boundary of local cluster. In other words, to enable communication among workers inside and outside of a cluster.
There are several public mqtt brokers for testing (not in any particular order):
broker.hivemq.com, mqtt.eclipse.org, test.mosquitto.org, broker.emqx.io, etc.

To configure a public mqtt broker, in fiab folder, update values.yaml file [here](../../fiab/helm-chart/values.yaml).
For example,
```yaml
broker:
  sort: mqtt
  host: broker.hivemq.com
```

After updating the file, one should start the whole system by running `flame.sh` (refer to [fiab](../../docs/03-fiab.md) for more details).
If the system was already started before the file modification, stop the whole system and start it again.

### Step 1: create a design
```
flamectl create design mnist_non_orchestration -d "mnist example for non-orchestration mode"
```

### Step 2: create a schema for design mnist
```
flamectl create schema schema.json --design mnist_non_orchestration
```

### Step 3: create (i.e., add) mnist code to the design

```
flamectl create code mnist.zip --design mnist_non_orchestration
```
Note: to understand relationship between schema and code, unzip mnist.zip and check the folder structure in it.

### Step 4: create a job
```
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
```
flamectl get jobs
```
For example,
```
$ flamectl get jobs
+--------------------------+-------+--------------------------------+-------------------------------+-------------------------------+
|          JOB ID          | STATE |           CREATED AT           |          STARTED AT           |           ENDED AT            |
+--------------------------+-------+--------------------------------+-------------------------------+-------------------------------+
| 624888fda8001d773e34de43 | ready | 2022-04-02 17:33:49.915 +0000  | 0001-01-01 00:00:00 +0000 UTC | 0001-01-01 00:00:00 +0000 UTC |
|                          |       | UTC                            |                               |                               |
+--------------------------+-------+--------------------------------+-------------------------------+-------------------------------+
$
```

### Step 5: check tasks

Also by running `flamectl get tasks <job_id>`, one can check the list of tasks created within the job.
For example,
```bash
$ flamectl get tasks 624888fda8001d773e34de43
+--------------------------+------------------------------------------+--------+-------+--------------------------------+
|          JOB ID          |                 AGENT ID                 |  TYPE  | STATE |           TIMESTAMP            |
+--------------------------+------------------------------------------+--------+-------+--------------------------------+
| 624888fda8001d773e34de43 | 18d2671fbf597f2200fd4a01f4dfc7878fce5ca9 | user   | ready | 2022-04-02 17:33:49.927 +0000  |
|                          |                                          |        |       | UTC                            |
| 624888fda8001d773e34de43 | 3e27750dfc40b95f853c11b691314899fdf2dfd1 | system | ready | 2022-04-02 17:33:49.932 +0000  |
|                          |                                          |        |       | UTC                            |
+--------------------------+------------------------------------------+--------+-------+--------------------------------+
$
```

In the above example, there is a task whose type is `user`. This task is one that can be executed by a participant, not by the Flame system.
In contrast, another task whose type is system is managed by the Flame system.

### Step 6: start user-type task

Let's first start the user-type task. In this example, we use docker image built in minikube VM and manually run a user docker container in the VM.
Therefore, the container is outside of the minikube cluster.

To run the user-type task, we need four parameters: apiserver IP, notifier IP, agent ID and agent key.
To obtain IP addresses, run the following command:
```
$ kubectl get services -n flame | grep  -e apiserver -e notifier
flame-apiserver                  LoadBalancer   10.105.11.65     10.105.11.65     10100:31121/TCP   27m
flame-notifier                   ClusterIP      10.108.126.46    <none>           10101/TCP         27m
$
```
In the example above, apiserver's IP is 10.105.11.65 and notifier's IP is 10.108.126.46.

For agent ID, it can be obtained from `flamectl get tasks <job_id>` command (see the example in Step 5).
In this example, it's `18d2671fbf597f2200fd4a01f4dfc7878fce5ca9`.

```bash
$ minikube ssh
$ docker run \
    -e FLAME_AGENT_ID=18d2671fbf597f2200fd4a01f4dfc7878fce5ca9 \
    -e FLAME_AGENT_KEY=any_key_chosen_by_user \
    flame:latest \
    /usr/bin/flamelet -a http://10.105.11.65:10100 -n 10.108.126.46:10101
```
`any_key_chosen_by_user` is a key that can be set by the user. The key must be reused in order to rejoin the same job later.


### Step 7: start a job

Assuming the id is `6131576d6667387296a5ada3`, run the following command to schedule a job.
```
flamectl start job 6131576d6667387296a5ada3
```

### Step 8: check progress

By running `flamectl get tasks <job_id>`, one can check the status of each task.

```bash
$ flamectl get tasks 624888fda8001d773e34de43
+--------------------------+------------------------------------------+--------+---------+--------------------------------+
|          JOB ID          |                 AGENT ID                 |  TYPE  |  STATE  |           TIMESTAMP            |
+--------------------------+------------------------------------------+--------+---------+--------------------------------+
| 624888fda8001d773e34de43 | 18d2671fbf597f2200fd4a01f4dfc7878fce5ca9 | user   | running | 2022-04-02 18:24:45.704 +0000  |
|                          |                                          |        |         | UTC                            |
| 624888fda8001d773e34de43 | 3e27750dfc40b95f853c11b691314899fdf2dfd1 | system | running | 2022-04-02 18:24:32.114 +0000  |
|                          |                                          |        |         | UTC                            |
+--------------------------+------------------------------------------+--------+---------+--------------------------------+
$
```

After some time later, run the same command above. If the job is successfully completed, one should see message like the following:
```bash
$ flamectl get tasks 624888fda8001d773e34de43
+--------------------------+------------------------------------------+--------+-----------+--------------------------------+
|          JOB ID          |                 AGENT ID                 |  TYPE  |   STATE   |           TIMESTAMP            |
+--------------------------+------------------------------------------+--------+-----------+--------------------------------+
| 624888fda8001d773e34de43 | 18d2671fbf597f2200fd4a01f4dfc7878fce5ca9 | user   | completed | 2022-04-02 18:25:14.237 +0000  |
|                          |                                          |        |           | UTC                            |
| 624888fda8001d773e34de43 | 3e27750dfc40b95f853c11b691314899fdf2dfd1 | system | completed | 2022-04-02 18:25:19.023 +0000  |
|                          |                                          |        |           | UTC                            |
+--------------------------+------------------------------------------+--------+-----------+--------------------------------+
$
```

Also, the command `flamectl get jobs` will return messages similar to:
```
$ flamectl get jobs
+--------------------------+-----------+--------------------------------+--------------------------------+--------------------------------+
|          JOB ID          |   STATE   |           CREATED AT           |           STARTED AT           |            ENDED AT            |
+--------------------------+-----------+--------------------------------+--------------------------------+--------------------------------+
| 624888fda8001d773e34de43 | completed | 2022-04-02 17:33:49.915 +0000  | 2022-04-02 18:24:30.759 +0000  | 2022-04-02 18:25:19.033 +0000  |
|                          |           | UTC                            | UTC                            | UTC                            |
+--------------------------+-----------+--------------------------------+--------------------------------+--------------------------------+
$
```
