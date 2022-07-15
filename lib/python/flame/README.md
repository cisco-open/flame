# A guide to run python library locally without engaging with the system components

## Environment Setup
We recommend setting up your environment with `conda`. This example is based on Ubuntu 22.04.
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

## Configuring Brokers
As the flame system uses MQTT brokers to exchange messages during federated learning, to run the python library locally, you could either 1) install a local MQTT broker 2) use a public MQTT broker. Here we'll illustrate the second option.

Go to any examples that you wish to run locally in `examples` directory, change the `host` from `"flame-mosquitto"` to `broker.hivemq.com` in the `config.json` files of both the trainer and aggregator.

## Running the Python Code

```bash
cd examples/mnist/trainer

python keras/main.py config.json
```

```bash
# Open another terminal
conda activate flame
cd examples/mnist/aggregator

python keras/main.py config.json
```