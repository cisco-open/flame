# Copyright 2023 Cisco Systems, Inc. and its affiliates
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
"""Runtime for Metric Collector."""

import logging
import time

logger = logging.getLogger(__name__)


def time_tasklet(func):
    """Decorator to time Tasklet.do() function"""

    def wrapper(*args, **kwargs):
        s = args[0]
        if s.composer.mc:
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()

            s.composer.mc.save("runtime", s.alias, end - start)
            s.composer.mc.save("starttime", s.alias, start)
            logger.debug(f"Runtime of {s.alias} is {end-start}")
            return result
        else:
            logger.debug("No MetricCollector; won't record runtime")
            return func(*args, **kwargs)

    return wrapper
