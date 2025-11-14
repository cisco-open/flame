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

def timer_decorator(func):
    """Decorator to time TopAggregator function and log round/data info."""
    def wrapper(*args, **kwargs):
        logger.debug("Inside timer_decorator wrapper")
        self = args[0]  # TopAggregator

        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        duration = end - start

        stage = getattr(self, "fwd_llm_stage", None)
        if stage:
            logger.info(
                f"[decorator] Runtime of {func.__name__}: {duration:.6f}s "
                f"(Round={stage.round_id}, DataId={stage.data_id}, Iter={stage.iteration}, TrainerId={stage.trainer_id})"
            )
        else:
            logger.info(
                f"[decorator] Runtime of {func.__name__}: {duration:.6f}s (no stage info)")
        return result

    return wrapper

class FwdLLMStage:
    """Lightweight metadata object for each federated round of FwdLLM."""

    def __init__(self, round_id, data_id, iteration, trainer_id=None):
        self.round_id = round_id
        self.data_id = data_id
        self.iteration = iteration
        self.trainer_id = trainer_id

    def __repr__(self):
        return f"FwdLLMStage(round={self.round_id}, data_id={self.data_id}, iter={self.iteration})"


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
            logger.info(f"Runtime of {s.alias} is {end-start}")
            return result
        else:
            logger.debug("No MetricCollector; won't record runtime")
            return func(*args, **kwargs)

    return wrapper
