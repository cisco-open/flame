# Flame Python library — environment setup & running examples

This is the single starting point for setting up a dev environment and running
the bundled examples (async_cifar10, async_google_speech, fwdllm). You do **not**
need the SDK / Ubuntu / macOS quickstart docs for this — they cover the same
env setup plus control-plane (Go) bits you don't need just to run examples.

## 1. Prerequisites

- Linux with `conda` (miniconda/anaconda) installed.
- Python 3.10+ (3.11 recommended) — created for you in step 2.
- A GPU is recommended for these examples but not required.
- An MQTT broker reachable on `localhost:1883` (the examples use `backend: mqtt`).
  `mosquitto` works; see step 3.

## 2. Set up the environment (one command)

From the repo root:

```bash
bash scripts/setup_env.sh flame      # creates a py3.11 conda env named "flame"
conda activate flame
```

Equivalent manual steps:

```bash
conda create -n flame python=3.11 -y
conda activate flame
pip install -e lib/python[examples,dev]
```

The single `[examples]` extra installs the full stack for **all** bundled
examples — vision (`torch`/`torchvision`), speech (`torchaudio`), and NLP
(`transformers` + the `adapters` add-on + `h5py`/`pandas`/`scikit-learn`).
The dependency source of truth is [`lib/python/setup.py`](setup.py); there is
no `requirements.txt`.

## 3. Start an MQTT broker

```bash
pgrep -x mosquitto >/dev/null || mosquitto -d   # no-op if one is already running
```

## 4. Run an example (10-trainer smoke tests)

Run from the **repo root** with the env activated.

### async_cifar10 — vision (YAML launcher)

```bash
python -m flame.launch.run_experiment \
    lib/python/examples/async_cifar10/expt_scripts_2026/felix_n10_alpha100_syn20_smoke.yaml
```

### fwdllm — NLP / DistilBERT forward-mode (YAML launcher)

Requires the agnews H5 data referenced in the smoke YAML
(`data_file_path` / `partition_file_path`); update those paths for your host
if needed.

```bash
python -m flame.launch.run_experiment \
    lib/python/examples/fwdllm/expt_scripts/fwdllm_n10_smoke.yaml
# siblings: fwdllm_plus_n10_smoke.yaml, fluxtune_n10_smoke.yaml
```

### async_google_speech — speech

Not yet migrated to the `flame.launch` YAML launcher. Run it via the example's
own scripts under `lib/python/examples/async_google_speech/` (start the
aggregator, then `trainer/<config_dir>/exec_100_trainers.sh`). It only needs
the env from step 2 (`torchaudio` is included).

## Troubleshooting

| Symptom | Fix |
| :--- | :--- |
| `ModuleNotFoundError: flame` | `pip install -e lib/python` (env not active or not installed) |
| Trainers/aggregator hang at startup | No MQTT broker — run `mosquitto -d` (step 3) |
| fwdllm: `FileNotFoundError` on `*.h5` | Fix the `data_file_path`/`partition_file_path` in the smoke YAML |

More detail on the launcher and the shared `_metadata/` bundle:
[`examples/README.md`](examples/README.md).
