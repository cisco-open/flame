#!/bin/bash
# Quick setup script for async_cifar10 example
# Usage: bash setup_env.sh <env_name> [flame_repo_path]

set -e  # Exit on error

# Check if environment name is provided
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Error: Environment name required"
    echo "Usage: bash setup_env.sh <env_name> [flame_repo_path]"
    echo "  env_name: Name for the conda environment"
    echo "  flame_repo_path: Path to flame repository (default: auto-detect)"
    echo "Example: bash setup_env.sh my_flame_env"
    echo "Example: bash setup_env.sh my_flame_env /home/user/flame"
    exit 1
fi

ENV_NAME="$1"
FLAME_REPO_PATH="${2:-}"

echo "=========================================="
echo "Flame async_cifar10 Environment Setup"
echo "Environment: ${ENV_NAME}"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed or not in PATH${NC}"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

echo -e "${YELLOW}Step 1: Creating conda environment '${ENV_NAME}' with Python 3.9${NC}"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}Environment '${ENV_NAME}' already exists. Skipping creation.${NC}"
else
    conda create -n ${ENV_NAME} python=3.9 -y
    echo -e "${GREEN}✓ Environment created${NC}"
fi

echo ""
echo -e "${YELLOW}Step 2: Activating environment${NC}"
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}
echo -e "${GREEN}✓ Environment activated${NC}"

echo ""
echo -e "${YELLOW}Step 3: Installing dependencies from requirements.txt${NC}"
# Determine Flame repository root directory
if [ -n "$FLAME_REPO_PATH" ]; then
    ROOT_DIR="$FLAME_REPO_PATH"
    echo "Using provided flame repo path: $ROOT_DIR"
else
    # Auto-detect: go up 4 levels from async_cifar10 directory
    ROOT_DIR="$(cd "$(dirname "$0")/../../../../" && pwd)"
    echo "Auto-detected flame repo path: $ROOT_DIR"
fi

if [ -f "$ROOT_DIR/requirements.txt" ]; then
    echo "Installing from: $ROOT_DIR/requirements.txt"
    pip install -r "$ROOT_DIR/requirements.txt"
    echo -e "${GREEN}✓ Dependencies installed from requirements.txt${NC}"
else
    echo -e "${RED}✗ requirements.txt not found at $ROOT_DIR${NC}"
    echo -e "${RED}Please provide the correct flame repository path as second argument${NC}"
    echo -e "${RED}Usage: bash setup_env.sh <env_name> <flame_repo_path>${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Step 4: Installing Flame library${NC}"
cd "$(dirname "$0")/../../"  # Navigate to lib/python
pip install -e .
echo -e "${GREEN}✓ Flame library installed${NC}"

echo ""
echo -e "${YELLOW}Step 6: Checking MQTT broker${NC}"
if systemctl is-active --quiet mosquitto 2>/dev/null; then
    echo -e "${GREEN}✓ MQTT broker (mosquitto) is running${NC}"
elif pgrep -x mosquitto > /dev/null 2>&1; then
    echo -e "${GREEN}✓ MQTT broker (mosquitto) is running${NC}"
elif command -v mosquitto &> /dev/null; then
    echo -e "${YELLOW}⚠ MQTT broker installed but not running${NC}"
    echo "Please contact your system administrator to start mosquitto"
else
    echo -e "${RED}⚠ MQTT broker not found${NC}"
    echo "Please contact your system administrator to install mosquitto"
fi

echo ""
echo -e "${YELLOW}Step 7: Checking GPU availability${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo -e "${GREEN}✓ Found ${GPU_COUNT} GPU(s)${NC}"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
else
    echo -e "${YELLOW}⚠ nvidia-smi not found. GPU support may not be available${NC}"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "To use the environment:"
echo "  1. conda activate ${ENV_NAME}"
echo "  2. export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$CONDA_PREFIX/lib/"
echo "  3. cd $ROOT_DIR/lib/python/examples/async_cifar10"
echo ""
echo "Quick test:"
echo "  cd expt_scripts_2026/scripts"
echo "  ./oort_n300_oracular_1feb_all4unavail.sh test_node"
echo ""
echo "Note: Ensure MQTT broker is running before starting experiments."
echo "For more details, see: lib/python/examples/async_cifar10/README.md"
