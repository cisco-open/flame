## Oort MNIST Example

We use MNIST for an example of Oort, which is a client (trainer) selection framework for federated learning proposed by:

> [OSDI'21](https://www.usenix.org/conference/osdi21/)
>
> [Oort: Efficient Federated Learning via Guided Participant Selection](https://www.usenix.org/conference/osdi21/presentation/lai)


This example runs within the `conda` environment.
After installing Flame SDK into `flame` environment, change directory to `lib/python/flame/examples/oort_mnist`, and run the following command.

```bash
$ conda activate flame
```

Oort supports any number of trainers.
For example, to run FL with Oort selector using 20 trainers and you want to select 4 trainers at a round, you can run:

```bash
$ python run.py {num_of_trainers_to_aggregate_at_a_round} {num_of_total_trainers},

# EXAMPLE:
# python run.py 4 20
```

You can track the progress by running the following commands:

```bash
$ cat output/aggregator/aggregator.txt
```

## Notes
- When you want to aggregate `aggr_num` trainers at a round, Oort selects `1.3 * aggr_num` trainers and aggregates `aggr_num` trainers' updates in the order of arrival. For example, if you desire to select 10 trainers, Oort selects 13 trainers and aggregates 10 fast trainers.
