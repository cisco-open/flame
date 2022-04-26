## MNIST in non-orchestration mode

Here we go over how to run MNIST example in non-orchestration mode within [fiab](../../docs/03-fiab.md) environment.
In non-orchestration mode, participants drive participation. Hence, participants should bring their dataset along with compute.

This example is almost identical to the mnist example shown [here](../mnist/README.md) with a few configuration changes.
To see differences, use `diff`. For example, do `diff schema.json ../mnist/schema.json`.

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
|          JOB ID          |                 TASK ID                  |  TYPE  | STATE |           TIMESTAMP            |
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

For task ID, it can be obtained from `flamectl get tasks <job_id>` command (see the example in Step 5).
In this example, it's `18d2671fbf597f2200fd4a01f4dfc7878fce5ca9`.

```bash
$ minikube ssh
$ docker run \
    -e FLAME_TASK_ID=18d2671fbf597f2200fd4a01f4dfc7878fce5ca9 \
    -e FLAME_TASK_KEY=any_key_chosen_by_user \
    --dns 10.96.0.10 \
    flame:latest \
    /usr/bin/flamelet \
    -a http://flame-apiserver.flame.svc.cluster.local:10100 \
    -n flame-notifier.flame.svc.cluster.local:10101
```
`any_key_chosen_by_user` is a key that can be set by the user. The key must be reused in order to rejoin the same job later.

The IP address (10.96.0.10) of a dns server is one for kube-dns. The IP address is unlikely to change.
You can check the IP with the following command:
```bash
$ kubectl get services -A | grep kube-dns
kube-system   kube-dns                         ClusterIP      10.96.0.10       <none>           53/UDP,53/TCP,9153/TCP   10d
```

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
|          JOB ID          |                 TASK ID                  |  TYPE  |  STATE  |           TIMESTAMP            |
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
|          JOB ID          |                 TASK ID                  |  TYPE  |   STATE   |           TIMESTAMP            |
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
