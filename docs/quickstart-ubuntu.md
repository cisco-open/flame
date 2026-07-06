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

From the cloned repo root:

```bash
bash scripts/setup_env.sh flame
conda activate flame
```

That creates a conda env named `flame` (Python 3.11) and installs the flame
library plus the `[examples]` and `[dev]` extras (torch, torchvision,
sortedcontainers, wandb, pytest, ...). The `[examples]` extra also pulls the
NLP forward-mode stack (transformers, adapters, h5py, ...), so one env runs
the smoke tests for both async_cifar10 and fwdllm. It uses the dependency spec
in `lib/python/setup.py` — there is no separate `requirements.txt`.

### Running an Example

Experiments are driven by YAML descriptors consumed by `flame.launch`. For
example, a 10-trainer Felix smoke test on async CIFAR-10:

```bash
python -m flame.launch.run_experiment \
    lib/python/examples/async_cifar10/expt_scripts_2026/felix_n10_alpha100_syn20_smoke.yaml
```

The launcher:
1. Parses the experiment YAML.
2. Merges the baseline (e.g. `baseline: felix`) + per-experiment overrides
   into a complete aggregator config; prints field provenance (which layer
   contributed each value).
3. Spawns the aggregator and N trainers via `--config-json`.
4. Writes logs + a snapshot of the merged configs into
   `experiments/run_<timestamp>_<name>/`.

For the catalog of available examples and details of the metadata
hierarchy (shared trainer registry, traces, dataset splits, baselines),
see [`lib/python/examples/README.md`](../lib/python/examples/README.md).
