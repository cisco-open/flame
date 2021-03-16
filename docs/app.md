# App specification

An ML app runs within a scope of namespace. In other words, all the ML components should be described within a namespace.
A namespace is an abstracted container for an ML app. Hence, a namespace must be globally unique.
A namespace consists of roles (ML components) and channels (link between components).
A role is an abstraction entity that interacts with the other role connected via a channel.
The incarnation of a role is specific to application.

## App

The follwoing shows a basic schema for an app.
```
type: app
namespace: <name>
priority:
  - trainingScore: <min_accuracy_or_score>
  - trainingTime: <max_training_time_in_minute>
  - trainingCost: <max_monthly_dollar_amount>
  - inferenceLatency: <max_latency_in_millisecond>
  - inferenceCost: <max_monthly_dollar_amount>

roles:
  - <role1>
  - <role2>
  ...

channels:
  - <channel1>:
      backend: <backend_name1>
    # birectional channel
    direction:
      - <role1> -> <role2>
      - <role2> -> <role1>
  - <channel2>:
    backend: <backend_name2>
    # peer-to-peer channel
    direction:
      - <role1> -> <role1>
  ...
```

With the skeleton schema above, many ML scenarios can be supported.
The following shows a typical federated learning scenario.
```
type: app
namespace: federated_learning
priority:
  - trainingScore: 0.95
  - trainingTime: 60
  - trainingCost: 1000
  - inferenceLatency: 100
  - inferenceCost: 10

roles:
  - aggregator
  - trainer
  
channels:
  - parameter_channel:
    backend: default
    direction:
      - aggregator -> trainer
      - trainer -> aggregator
```

Distributed ML scenario can be expressed as follows.
```
type: app
namespace: distributed_learning
roles:
  - trainer

channels:
  - peer_channel:
    backend: mpi
    direction:
      - trainer -> trainer
```

A hybrid scenario combined with federated and distributed learning looks like the following.
```
type: app
namespace: hybrid_learning
roles:
  - aggregator
  - trainer

channels:
  - parameter_channel:
    backend: default
    direction:
      - aggregator -> trainer
      - trainer -> aggregator

  - peer_channel:
    backend: mpi
    direction:
      - trainer -> trainer
```

## Role

The following shows a skeleton of role specification.
```
type: role
name: <role_name>
provision:
  type: <provision_type> # static, dynamic
  count: <number_of_instances> # applied only for static provision type
  location: # applied only for static provision type
    provider: <provider>
	region: <region>

configuration:
  cpu: 
	min: <no_of_cpu_cores>
	max: <no_of_cpu_cores>
  gpu:
	min: <no_of_gpu_cores>
	max: <no_of_gpu_cores>
  mem:
	min: <memory_size_in_mb>
	max: <memory_size_in_mb>
  disk:
    min: <disk_size_in_gb>
	max: <disk_size_in_gb>
```
When provision type is specified as `dynamic`, instances for the role operate in runtime environments
under control of different authorities. Then, those instances can join and leave the app (i.e, the app's namespace) over time.
For example, the `dynamic` provision type is effective when a trainer will run on edge devices whose owner is not
an ML app engineer. In case of `dynamic` provision, `count` and `location` properties are ignored.
The `resources` property may not be respected depending on runtime environment's state.

The following example shows a specification for aggregator role.
```
type: role
name: aggregator
provision:
  type: static
  count: 1
  location:
	provider: aws
	region: us-west-1

resources:
  cpu: 
	min: 2
	max: 4
  gpu:
	min: 0
	max: 0
  mem:
	min: 4096
	max: 8096
  disk:
    min: 128
	max: 128
```

The following shows an example specification for trainer role.
```
type: role
name: trainer
provision:
  type: dynamic

configuration:
  cpu: 
	min: 1
	max: 1
  gpu:
	min: 1
	max: 1
  mem:
	min: 4096
	max: 8096
  disk:
    min: 128
	max: 128
```
