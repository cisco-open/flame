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

The following brokers are all for local testing.
If you wish to run federated learning accross multiple machines, please consider using the MQTT public broker.
This means setting `backend` and the `mqtt` broker in the config file as follows:

```json
    "backend": "mqtt",
    "brokers": [
        {
            "host": "broker.hivemq.com",
            "sort": "mqtt"
        }
    ]
```

However, this may lead to job ID collisions since it is a public broker.
Thus, for local testing, we recommend using either of the two options below.

#### Local MQTT Broker

Since the flame system uses MQTT brokers to exchange messages during federated learning, to run the python library locally, you may install a local MQTT broker as shown below.

```bash
sudo apt update
sudo apt install -y mosquitto
sudo systemctl status mosquitto
```

The last command should display something similar to this:

```bash
mosquitto.service - Mosquitto MQTT v3.1/v3.1.1 Broker
     Loaded: loaded (/lib/systemd/system/mosquitto.service; enabled; vendor pre>
     Active: active (running) since Fri 2023-02-03 14:05:55 PST; 1h 20min ago
       Docs: man:mosquitto.conf(5)
             man:mosquitto(8)
   Main PID: 75525 (mosquitto)
      Tasks: 3 (limit: 9449)
     Memory: 1.9M
     CGroup: /system.slice/mosquitto.service
             └─75525 /usr/sbin/mosquitto -c /etc/mosquitto/mosquitto.conf
```

That confirms that the mosquitto service is active.
From now on, you may use `sudo systemctl stop mosquitto` to stop the mosquitto service, `sudo systemctl start mosquitto` to start the service, and `sudo systemctl restart mosquitto` to restart the service.

Go ahead and change the two config files in `mnist/trainer` and `mnist/aggregator` to make sure `backend` is `mqtt`.

```json
    "backend": "mqtt",
    "brokers": [
        {
            "host": "localhost",
            "sort": "mqtt"
        },
	{
	    "host": "localhost:10104",
	    "sort": "p2p"
	}
    ]
```

Note that if you also want to use the local `mqtt` broker for other examples you should make sure that the `mqtt` broker has `host` set to `localhost`.

#### P2P

To start a `p2p` broker, go to the top `/flame` directory and run:

```bash
make install
cd ~
sudo ./.flame/bin/metaserver
```

After changing the two config files in `mnist/trainer` and `mnist/aggregator` so that `backend` is set to `p2p`, continue to the next section.

### Running an Example

In order to run this example, you will need to open two terminals.

In the first terminal, run the following commands:

```bash
conda activate flame
cd ../examples/mnist/trainer

python keras/main.py config.json
```

Open another terminal and run:

```bash
conda activate flame
cd ../examples/mnist/aggregator

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
