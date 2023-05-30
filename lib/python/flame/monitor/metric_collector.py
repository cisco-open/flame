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
"""Metric Collector."""

import logging
import os
import time
import threading
import psutil
import numpy as np
from collections import defaultdict
from gpustat.nvml import pynvml as N

logger = logging.getLogger(__name__)


class MetricCollector:
    def __init__(self):
        """Initialize Metric Collector."""
        self.state_dict = dict()
        self.stat_log = defaultdict(list)

        # CPU monitoring
        cpu_thread = threading.Thread(target=self.gather_cpu_stats)
        cpu_thread.daemon = True
        cpu_thread.start()

        # GPU monitoring
        gpu_thread = threading.Thread(target=self.gather_gpu_stats)
        gpu_thread.daemon = True
        gpu_thread.start()

    def gather_gpu_stats(self, interval=1):
        pid = os.getpid()

        try:
            N.nvmlInit()
        except:
            logger.debug(
                "could not start gpustat.nvml.pynvml; no GPU metrics will be collected"
            )
            return

        dcount = N.nvmlDeviceGetCount()
        if dcount == 0:
            logger.debug("no GPUs detected")
            return

        while True:
            for d in range(dcount):
                try:
                    handle = N.nvmlDeviceGetHandleByIndex(d)
                    processes = N.nvmlDeviceGetComputeRunningProcesses(handle)
                except:
                    logger.debug(f"failed to retrieve GPU processes for GPU {d}")
                    continue

                curr_pids = [process.pid for process in processes]

                # GPU utilization
                # TO DO: implement metric gathering for process-specific utilization of the GPUs
                try:
                    if pid in curr_pids:
                        self.stat_log[f"gpu{d}_utilization"].append(
                            N.nvmlDeviceGetUtilizationRates(handle).gpu
                        )
                    else:
                        self.stat_log[f"gpu{d}_utilization"].append(0)
                except:
                    logger.debug(f"failed to get GPU utilization of GPU {d}")

                # GPU memory usage of process
                total_mem = 0
                for process in processes:
                    if pid == process.pid:
                        total_mem += process.usedGpuMemory

                self.stat_log[f"gpu{d}_memory"].append(total_mem)

            time.sleep(interval)

    def gather_cpu_stats(self, interval=1):
        pid = os.getpid()
        proc = psutil.Process(pid)

        while True:
            self.stat_log["cpu_utilization"].append(proc.cpu_percent())

            mem_info = proc.memory_info()
            self.stat_log["cpu_memory_rss"].append(mem_info.rss)
            self.stat_log["cpu_memory_vsz"].append(mem_info.vms)

            time.sleep(interval)

    def get_key(self, mtype, alias):
        return f"{alias}.{mtype}"

    def save(self, mtype, alias, value):
        """Saves key-value pair for metric."""
        key = self.get_key(mtype, alias)
        self.state_dict[key] = value
        logger.debug(f"Saving state_dict[{key}] = {value}")

    def accumulate(self, mtype, alias, value):
        key = self.get_key(mtype, alias)
        self.state_dict[key] = value + self.state_dict.get(key, 0)
        logger.debug(f"Accumulating metric state_dict[{key}] = {self.state_dict[key]}")

    def save_log_statistics(self):
        for name in self.stat_log:
            values = self.stat_log[name]
            if values:
                self.save(name, "min", np.min(values))
                self.save(name, "max", np.max(values))
                self.save(name, "mean", np.mean(values))
                self.save(name, "25th_prcntl", np.percentile(values, 25))
                self.save(name, "50th_prcntl", np.percentile(values, 50))
                self.save(name, "75th_prcntl", np.percentile(values, 75))
                self.save(name, "std", np.std(values))

    def get(self):
        """Returns the current metrics that were collected."""
        self.save_log_statistics()
        return self.state_dict

    def clear(self):
        self.state_dict = dict()
        for key in self.stat_log:
            self.stat_log[key].clear()
