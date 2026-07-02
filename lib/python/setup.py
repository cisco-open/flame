# Copyright 2022 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Package setup script."""

from setuptools import find_packages, setup

setup(
    name="flame",
    version="0.1.1",
    author="Flame Maintainers",
    author_email="flame-github-owners@cisco.com",
    include_package_data=True,
    packages=find_packages(),
    data_files=[],
    scripts=["scripts/flame-config"],
    url="https://github.com/cisco-open/flame/",
    license="LICENSE.txt",
    description="Python library to run ML workloads in the flame system",
    long_description=open("README.md").read(),
    python_requires=">=3.10",
    # Core flame package deps. Example/test/dev deps live in extras.
    install_requires=[
        "aiostream",
        "boto3",
        "cloudpickle",
        "diskcache",
        "fedscale",
        "gpustat",
        "grpcio",
        "mlflow",
        "numpy",
        "paho-mqtt",
        "protobuf",
        "psutil",
        "pydantic",
        "PyYAML",
        "requests",
        "shared-memory-dict",
        "zstandard",
    ],
    extras_require={
        # Runtime deps for examples/<x>/ workloads. Installing this one extra
        # is enough to run the smoke tests for BOTH the vision examples
        # (async_cifar10, ...) and the NLP forward-mode examples (fwdllm).
        "examples": [
            # --- Shared (vision + speech + NLP) ---
            # On CUDA driver >= 12.9 use setup_env.sh (force-reinstalls torch
            # from the cu126 index) instead of a bare `pip install -e .[examples]`.
            "torch",
            "torchvision",
            "torchaudio",  # async_google_speech (audio feature extraction)
            "sortedcontainers",
            "wandb",
            # --- fwdllm: DistilBERT forward-mode FL on agnews ---
            # Modern mainline `transformers` + the standalone `adapters` add-on
            # (the maintained successor to the un-buildable `adapter-transformers`
            # fork; both ship prebuilt wheels, so no Rust/tokenizers compile).
            # `adapters` pins `transformers` in lockstep (1.3.x requires
            # transformers ~=4.57.6), so they are versioned together here.
            # Verified set: transformers 4.57.6 / adapters 1.3.0 / tokenizers
            # 0.22.2 / h5py 3.16.0 on torch 2.x + numpy 2.x + py3.11.
            "transformers>=4.57,<4.58",
            "adapters>=1.3,<1.4",
            "h5py>=3",
            "pandas>=2",
            "scikit-learn>=1.3",
            "setproctitle",
        ],
        "dev": [
            "pytest",
            "pre-commit",
            "black",
            "flake8",
            "bandit",
            "mypy",
            "isort",
        ],
    },
)
