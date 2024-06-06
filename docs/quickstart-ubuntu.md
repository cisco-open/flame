## Quickstart

### Prerequisites

* Ubuntu 20.04 or 22.04
* Install [anaconda](https://www.anaconda.com/download/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) in order to create the environment.
* Clone repo (you could use `git clone https://github.com/cisco-open/flame.git`).

### Local MQTT Broker

Since the flame system uses an MQTT broker to exchange messages during federated learning, to run the python library locally, you may install a local MQTT broker as shown below.

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

You can use the following commands to stop and start the mosquitto service:

```bash
# start mosquitto
sudo systemctl start mosquitto
# stop mosquitto
sudo systemctl stop mosquitto
# restart mosquitto
sudo systemctl restart mosquitto
```

Go ahead and change the two config files `flame/lib/python/examples/mnist/trainer/config.json` and `flame/lib/python/examples/mnist/aggregator/config.json` to set `backend` to `mqtt`.

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

### Environment Setup

We recommend setting up your environment with `conda`. Within the cloned flame directory, run the following to activate and setup the flame environment:

```bash
# Run within the cloned flame directory
cd lib/python/flame
conda create -n flame python=3.9
conda activate flame

pip install google
pip install tensorflow
pip install torch
pip install torchvision

cd ..
make install
```

To install flame in editable development mode, run the following instead of the `make install` command:
```bash
python -m pip install -e .
```

### Running an Example

We will run the MNIST example with one aggregator and one trainer.

Open two terminal windows.

In the first terminal, once you are in `flame/lib/python/examples/mnist/trainer`, run:

```bash
conda activate flame

python keras/main.py config.json
```

Open another terminal in `flame/lib/python/examples/mnist/aggregator` and run:

```bash
conda activate flame

python keras/main.py config.json
```

In this example, we have one aggregator and one trainer that runs with the same job ID and different task IDs.
After running, you will see the aggregator (second terminal) sending a global model to the trainer (first terminal), and the trainer sending the updated local model back to the aggregator.

This completes one round of communication between the aggregator and trainer.

The current example is set to 20 rounds (see the `hyperparameters` section of the `flame/lib/python/examples/mnist/aggregator/config.json` file), meaning the communication protocol described earlier will repeat 20 times.
