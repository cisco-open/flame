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
        # Runtime deps for examples/<x>/ workloads (torch + utilities).
        "examples": [
            "torch",
            "torchvision",
            "sortedcontainers",
            "wandb",
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
