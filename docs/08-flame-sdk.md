# Flame SDK

## Environment Setup
We recommend setting up your environment with `conda`. This example is based on Ubuntu 22.04. Run the following inside of the `lib/python/flame` directory.

```bash
conda create -n flame python=3.9
conda activate flame


pip install google
pip install tensorflow
pip install torch
pip install torchvision

cd ..
make install
```

## Quickstart

### Configuring Brokers
As the flame system uses MQTT brokers to exchange messages during federated learning, to run the python library locally, you could either 1) install a local MQTT broker 2) use a public MQTT broker. Here we'll illustrate the second option.

Go to any examples that you wish to run locally in `examples` directory, change the `host` from `"flame-mosquitto"` to `broker.hivemq.com` in the `config.json` files of both the trainer and aggregator.

### Running an Example

In order to run this example, you will need to open two terminals.

In the first terminal, run the following command:

```bash
cd examples/mnist/trainer

python keras/main.py config.json
```

Open another terminal and run:

```bash
conda activate flame
cd examples/mnist/aggregator

python keras/main.py config.json
```

## Configuration

### Selector
Users are able to implement new selectors in `lib/python/flame/selector/` which should return a dictionary with keys corresponding to the active trainer IDs (i.e., agent IDs). After implementation, the new selector needs to be registered into both `lib/python/flame/selectors.py` and `lib/python/flame/config.py`.

#### Currently Implemented Selectors
1. Naive (i.e., select all)
```json
"selector": {
    "sort": "default",
    "kwargs": {}
}
```
2. Random (i.e, select k out of n local trainers)
```json
"selector": {
    "sort": "random",
    "kwargs": {
        "k": 1
    }
}
```

### Optimizer (i.e., aggregator of FL)
Users can implement new server optimizer, when the client optimizer is defined in the actual ML code, in `lib/python/flame/optimizer` which can take in hyperparameters if any and should return the aggregated weights in either PyTorch of Tensorflow format. After implementation, the new optimizer needs to be registered into both `lib/python/flame/optimizer.py` and `lib/python/flame/config.py`.

#### Currently Implemented Optimizers
1. FedAvg (i.e., weighted average in terms of dataset size)
```json
# e.g.
"optimizer": {
    "sort": "fedavg",
    "kwargs": {}
}
```
2. FedAdaGrad (i.e., server uses AdaGrad optimizer)
```json
"optimizer": {
    "sort": "fedadagrad",
    "kwargs": {
        "beta_1": 0,
        "eta": 0.1,
        "tau": 0.01
    }
}
```
3. FedAdam (i.e., server uses Adam optimizer)
```json
"optimizer": {
    "sort": "fedadam",
    "kwargs": {
        "beta_1": 0.9,
        "beta_2": 0.99,
        "eta": 0.01,
        "tau": 0.001
    }
}
```
4. FedYogi (i.e., servers use Yogi optimizer)
```json
"optimizer": {
    "sort": "fedyogi",
    "kwargs": {
        "beta_1": 0.9,
        "beta_2": 0.99,
        "eta": 0.01,
        "tau": 0.001
    }
}
```
