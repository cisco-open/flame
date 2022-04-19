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
from typing import List
from ..config import Config

from pip._internal.cli.main import main as pipmain

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
        for val in v.groupby.value:
            if val in config.realm:
                groupby_value = groupby_value + val + "-" 

    return config.role + '-' + groupby_value + config.agent_id[:8]
