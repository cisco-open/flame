# Copyright (c) 2021 Cisco Systems, Inc. and its affiliates
# All rights reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import concurrent.futures
from contextlib import contextmanager
from threading import Thread
from typing import List

from pip._internal.cli.main import main as pipmain


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
