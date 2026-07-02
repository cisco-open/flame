#!/bin/bash
# One-shot conda env setup for flame + examples.
# Usage: bash scripts/setup_env.sh <env_name>

set -e

if [ "$#" -lt 1 ]; then
  echo "usage: bash scripts/setup_env.sh <env_name>"
  exit 1
fi
ENV_NAME="$1"
PY_VERSION="3.11"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v conda &> /dev/null; then
  echo "error: conda not found. install miniconda or anaconda first." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "env '$ENV_NAME' already exists; activating + reinstalling flame."
else
  conda create -y -n "$ENV_NAME" "python=$PY_VERSION"
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

pip install --upgrade pip
pip install -e "$REPO_ROOT/lib/python[examples,dev]"

# CUDA 12.6 wheels are required for driver >= 12.9 compatibility.
# The default PyPI torch wheel is built against an older CUDA runtime and will
# silently misdetect or error on CUDA driver 12.9+.  Force-reinstall from the
# cu126 index to get the matching runtime wheel.
pip install --force-reinstall torch==2.12.0 torchvision==0.27.0 \
  --index-url https://download.pytorch.org/whl/cu126

# Optional: warn if mosquitto broker isn't around.
if ! command -v mosquitto &> /dev/null; then
  echo "warning: mosquitto MQTT broker not installed. examples will need one running on localhost:1883."
fi

echo ""
echo "done. activate with:  conda activate $ENV_NAME"
echo "run tests with:       cd lib/python && python -m pytest tests/"
echo "run a smoke expt:     python -m flame.launch.run_experiment <yaml>"
