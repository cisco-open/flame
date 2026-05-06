# Copyright 2023 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#
# SPDX-License-Identifier: Apache-2.0
"""DynamicKCController: manages dynamic updates to K and C during FL training."""

import logging
from typing import Tuple

from flame.selector.dynamic_kc_policy import DynamicKCPolicy

logger = logging.getLogger(__name__)

_SUMMARY_METRIC_KEYS = [
    "var_pass_rate",
    "var_last",
    "avg_staleness",
    "n_eligible_train",
    "n_eligible_eval",
    "model_version",
]


def _fmt_metrics(metrics: dict) -> dict:
    """Return a truncated metrics dict suitable for a single log line."""
    out = {}
    for k in _SUMMARY_METRIC_KEYS:
        if k in metrics:
            v = metrics[k]
            out[k] = round(v, 3) if isinstance(v, float) else v
    return out


class DynamicKCController:
    """Controls dynamic updates to K (aggregation goal) and C (concurrency).

    The aggregator calls ``step(metrics)`` after each aggregation; the controller
    clamps policy output to [k_min, k_max] / [c_min, c_max] and returns (k, c).
    C is propagated to the selector via ``channel.set_property("dynamic_c", c)``.
    """

    def __init__(
        self,
        policy: DynamicKCPolicy,
        k_init: int,
        c_init: int,
        k_min: int,
        k_max: int,
        c_min: int,
        c_max: int,
        update_every_n_aggs: int = 1,
    ):
        self.policy = policy
        self.k = k_init
        self.c = c_init
        self.k_min = k_min
        self.k_max = k_max
        self.c_min = c_min
        self.c_max = c_max
        self.update_every_n_aggs = update_every_n_aggs

        self._agg_counter = 0
        self._total_updates = 0
        self._k_history: list = [(0, k_init)]
        self._c_history: list = [(0, c_init)]

    def step(self, metrics: dict) -> Tuple[int, int]:
        """Process one aggregation and return (k, c) after applying policy."""
        self._agg_counter += 1
        self._total_updates += 1

        if self._agg_counter < self.update_every_n_aggs:
            return self.k, self.c

        self._agg_counter = 0

        new_k = self.policy.compute_new_k(self.k, metrics)
        new_c = self.policy.compute_new_c(self.c, metrics)

        if new_k is not None:
            # Enforce that K cannot exceed C, nor k_max
            upper_bound = min(self.k_max, self.c)
            clamped = max(self.k_min, min(upper_bound, new_k))
            if clamped != self.k:
                logger.info(
                    f"[DynamicKC] K: {self.k} → {clamped} "
                    f"(policy={self.policy.name()}, "
                    f"metrics={_fmt_metrics(metrics)})"
                )
                self.k = clamped
                self._k_history.append((self._total_updates, self.k))

        if new_c is not None:
            clamped = max(self.c_min, min(self.c_max, new_c))
            if clamped != self.c:
                logger.info(
                    f"[DynamicKC] C: {self.c} → {clamped} "
                    f"(policy={self.policy.name()}, "
                    f"metrics={_fmt_metrics(metrics)})"
                )
                self.c = clamped
                self._c_history.append((self._total_updates, self.c))

        return self.k, self.c

    def get_k(self) -> int:
        return self.k

    def get_c(self) -> int:
        return self.c

    def summary(self) -> dict:
        return {
            "k": self.k,
            "c": self.c,
            "total_updates": self._total_updates,
            "k_changes": len(self._k_history) - 1,
            "c_changes": len(self._c_history) - 1,
            "k_history_last5": list(self._k_history[-5:]),
            "c_history_last5": list(self._c_history[-5:]),
            "policy": self.policy.name(),
        }
