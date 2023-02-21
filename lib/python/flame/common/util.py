# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""Utility functions."""

import asyncio
import concurrent.futures
import sys
from contextlib import contextmanager
from enum import Enum
from threading import Thread
from typing import List, Union

from pip._internal.cli.main import main as pipmain

from ..config import Config
from .typing import ModelWeights
from .constants import DeviceType

PYTORCH = 'torch'
TENSORFLOW = 'tensorflow'


class MLFramework(Enum):
    """Supported ml framework."""

    UNKNOWN = 1
    PYTORCH = 2
    TENSORFLOW = 3


ml_framework_in_use = MLFramework.UNKNOWN
valid_frameworks = [
    framework.name.lower() for framework in MLFramework
    if framework != MLFramework.UNKNOWN
]


def determine_ml_framework_in_use():
    """Determine which ml framework in use."""
    global ml_framework_in_use

    if PYTORCH in sys.modules:
        ml_framework_in_use = MLFramework.PYTORCH
    elif TENSORFLOW in sys.modules:
        ml_framework_in_use = MLFramework.TENSORFLOW


def get_ml_framework_in_use():
    """Return ml framework in use.

    Caveat: This function should be called after ml framework package or module
            is imported. Otherwise, this function will always return unknown
            framework type. Also, once the ml framework is identified, the type
            won't change for the rest of run time.
    """
    global ml_framework_in_use

    if ml_framework_in_use == MLFramework.UNKNOWN:
        determine_ml_framework_in_use()

    return ml_framework_in_use

def get_params_detached_pytorch(model):
    """Return copy of parameters of pytorch model disconnected from graph."""
    return [param.detach().clone() for param in model.parameters()]

@contextmanager
def background_thread_loop():

    def run_forever(loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    _loop = asyncio.new_event_loop()

    _thread = Thread(target=run_forever, args=(_loop, ), daemon=True)
    _thread.start()
    yield _loop


def run_async(coro, loop, timeout=None):
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return fut.result(timeout), True
    except concurrent.futures.TimeoutError:
        return None, False


def install_packages(packages: List[str]) -> None:
    for package in packages:
        if not install_package(package):
            print(f'Failed to install package: {package}')


def install_package(package: str) -> bool:
    if pipmain(['install', package]) == 0:
        return True

    return False


def mlflow_runname(config: Config) -> str:
    groupby_value = ""
    for v in config.channels.values():
        for val in v.group_by.value:
            if val in config.realm:
                groupby_value = groupby_value + val + "-"

    return config.role + '-' + groupby_value + config.task_id[:8]


def delta_weights_pytorch(a: ModelWeights,
                          b: ModelWeights) -> Union[ModelWeights, None]:
    """Return delta weights for pytorch model weights."""
    if a is None or b is None:
        return None

    return {x: a[x] - b[y] for (x, y) in zip(a, b)}


def delta_weights_tensorflow(a: ModelWeights,
                             b: ModelWeights) -> Union[ModelWeights, None]:
    """Return delta weights for tensorflow model weights."""
    if a is None or b is None:
        return None

    return [x - y for (x, y) in zip(a, b)]


def get_pytorch_device(dtype: DeviceType):
    import torch
    if dtype == DeviceType.CPU:
        device_name = "cpu"
    elif dtype == DeviceType.GPU:
        device_name = "cuda"
    else:
        raise TypeError(f"Device type {dtype} is not supported.")
    
    return torch.device(device_name)

def weights_to_device(weights, dtype: DeviceType):
    """Send model weights to device type dtype."""
    
    framework = get_ml_framework_in_use()
    if framework == MLFramework.TENSORFLOW:
        return weights
    elif framework == MLFramework.PYTORCH:
        torch_device = get_pytorch_device(dtype)
        return {name: weights[name].to(torch_device) for name in weights}
    
    return None

def weights_to_model_device(weights, model):
    """Send model weights to same device as model"""
    framework = get_ml_framework_in_use()
    if framework == MLFramework.TENSORFLOW:
        return weights
    elif framework == MLFramework.PYTORCH:
        # make assumption all tensors are on same device
        torch_device = next(model.parameters()).device
        return {name: weights[name].to(torch_device) for name in weights}
    
    return None
