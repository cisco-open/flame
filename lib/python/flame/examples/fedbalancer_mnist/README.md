## FedBalancer MNIST Example

We use MNIST for an example of FedBalancer, which is a trainer (client) sample selection framework for acclerated federated learning proposed by:

> [MobiSys'22](https://www.sigmobile.org/mobisys/2022/)
>
> [FedBalancer: Data and Pace Control for Efficient Federated Learning on Heterogeneous Clients
](https://arxiv.org/abs/2201.01601)


This example runs within the `conda` environment.
After installing Flame SDK into `flame` environment, change directory to `lib/python/flame/examples/fedbalancer_mnist`, and run the following command.

```bash
$ conda activate flame
```

When running FedBalancer, you need to specify the number of trainers, and FedBalancer parameters `{w, lss, dss, p, noise_factor}`.
Please refer to FedBalancer paper for more details about the parameters.
The recommended set of parameters are: `{w, lss, dss, p, noise_factor} = {20, 0.05, 0.05, 1.0, 0.0}`

To run FL with FedBalancer using 20 trainers and recommended set of parameters, you can run:

```bash
$ python run.py {num_of_total_trainers} {w} {lss} {dss} {p} {noise_factor}

# EXAMPLE:
# python run.py 20 20 0.05 0.05 1.0 0.0
```

You can track the progress by running the following commands:

```bash
$ cat output/aggregator/aggregator.log
```

