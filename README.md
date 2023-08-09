<p align="center"><img src="docs/images/logo.png" alt="flame logo" width="200"/></p>

[![](https://img.shields.io/badge/Flame-Join%20Slack-brightgreen)](https://join.slack.com/t/flame-slack/shared_invite/zt-1mprreo9z-FmpGb1UPi43JOFJKyhIqAQ)

Flame is a platform that enables developers to compose and deploy federated learning (FL) training workloads easily.
The system is comprised of a service (control plane) and a python library (data plane).
The service manages machine learning workloads, while the python library facilitates composition of ML workloads.
And the library is also responsible for executing FL workloads.
With extensibility of its library, Flame can support various experimentations and use cases.

## Environment Setup
We recommend setting up your environment with `conda`. Run the following inside of the `flame/lib/python/flame` directory.

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

### Local MQTT Broker

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

Go ahead and change the two config files in `flame/lib/python/examples/mnist/trainer` and `flame/lib/python/examples/mnist/aggregator` to make sure `backend` is `mqtt`.

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

### Running an Example

In order to run this example, you will need to open two terminals.

In the first terminal, once you are in `flame/lib/python/examples/mnist/trainer`, run the following commands:

```bash
conda activate flame

python keras/main.py config.json
```

Open another terminal in `flame/lib/python/examples/mnist/aggregator` and run:

```bash
conda activate flame

python keras/main.py config.json
```

You will see the aggregator (second terminal) sending a global model to the trainer (first terminal), and the trainer sending the updated local model back to the aggregator!
