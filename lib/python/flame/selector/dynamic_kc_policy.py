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
"""Dynamic K and C policies for FwdLLM federated learning."""

import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


class DynamicKCPolicy(ABC):
    """Abstract base for dynamic K (aggregation goal) and C (concurrency) policies.

    Policies are called by DynamicKCController after each aggregation.
    Each method returns a new integer value, or None to leave the current
    value unchanged.

    The ``metrics`` dict may contain any of the following keys (all optional;
    policies should use .get() and handle None gracefully):

        var_pass_rate     float [0,1]  Fraction of recent aggs where variance check passed
        var_last          float        Variance value from the most recent aggregation
        avg_staleness     float        Mean staleness of updates in the last K-window
        p75_staleness     float        75th-pct staleness
        model_version     int          Current model version
        n_aggs_completed  int          Total aggregations since run start
        var_threshold     float        FedSGDAggregator's current variance threshold
        n_eligible_train  int          Ends currently eligible for train task
        n_eligible_eval   int          Ends currently eligible for eval task
    """

    @abstractmethod
    def compute_new_k(self, current_k: int, metrics: dict) -> Optional[int]:
        """Return a new K value, or None to keep the current value."""

    @abstractmethod
    def compute_new_c(self, current_c: int, metrics: dict) -> Optional[int]:
        """Return a new C value, or None to keep the current value."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable name used in log messages."""


# ---------------------------------------------------------------------------
# Concrete policies
# ---------------------------------------------------------------------------


class NoOpPolicy(DynamicKCPolicy):
    """Returns None for both K and C — equivalent to static K and C.

    Use this as a baseline or during testing to verify that the controller
    plumbing is wired correctly without changing any behavior.
    """

    def compute_new_k(self, current_k: int, metrics: dict) -> Optional[int]:
        return None

    def compute_new_c(self, current_c: int, metrics: dict) -> Optional[int]:
        return None

    def name(self) -> str:
        return "noop"


class VarianceBasedPolicy(DynamicKCPolicy):
    """Adjusts K based on a rolling variance pass-rate window.

    - If the rolling pass-rate rises above ``high_threshold``: variance checks
      are passing consistently, meaning the model is receiving enough gradient
      diversity at the current K.  Decrease K by ``k_step`` to speed up iteration.
    - If the rolling pass-rate falls below ``low_threshold``: gradients are too
      noisy.  Increase K by ``k_step`` to collect more gradient diversity before
      each model update.
    - Within [low_threshold, high_threshold]: no change.

    This policy never touches C.
    """

    def __init__(
        self,
        high_threshold: float = 0.8,
        low_threshold: float = 0.3,
        k_step: int = 2,
        window: int = 10,
    ):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.k_step = k_step
        self._history: deque = deque(maxlen=window)

    def compute_new_k(self, current_k: int, metrics: dict) -> Optional[int]:
        rate = metrics.get("var_pass_rate")
        if rate is None:
            return None
        self._history.append(rate)
        if len(self._history) < self._history.maxlen:
            return None  # wait for a full window before acting
        avg = sum(self._history) / len(self._history)
        if avg > self.high_threshold:
            logger.debug(
                f"[VarianceBasedPolicy] avg_pass_rate={avg:.3f} > {self.high_threshold}: "
                f"decreasing K by {self.k_step}"
            )
            return current_k - self.k_step
        if avg < self.low_threshold:
            logger.debug(
                f"[VarianceBasedPolicy] avg_pass_rate={avg:.3f} < {self.low_threshold}: "
                f"increasing K by {self.k_step}"
            )
            return current_k + self.k_step
        return None

    def compute_new_c(self, current_c: int, metrics: dict) -> Optional[int]:
        return None

    def name(self) -> str:
        return "variance_based"


class StalenessBasedPolicy(DynamicKCPolicy):
    """Adjusts C based on rolling average staleness of received updates.

    High staleness means trainers are returning gradients computed on old model
    versions — a symptom of having too many concurrent trainers (high C).
    Decreasing C means each batch of selected trainers turns around faster,
    producing fresher gradients.

    - ``avg_staleness > stale_threshold``: too stale → decrease C by ``c_step``.
    - ``avg_staleness < fresh_threshold``: very fresh → can increase C.
    - Within [fresh_threshold, stale_threshold]: no change.

    This policy never touches K.
    """

    def __init__(
        self,
        stale_threshold: float = 3.0,
        fresh_threshold: float = 1.0,
        c_step: int = 5,
        window: int = 10,
    ):
        self.stale_threshold = stale_threshold
        self.fresh_threshold = fresh_threshold
        self.c_step = c_step
        self._history: deque = deque(maxlen=window)

    def compute_new_k(self, current_k: int, metrics: dict) -> Optional[int]:
        return None

    def compute_new_c(self, current_c: int, metrics: dict) -> Optional[int]:
        staleness = metrics.get("avg_staleness")
        if staleness is None:
            return None
        self._history.append(staleness)
        if len(self._history) < self._history.maxlen:
            return None
        avg = sum(self._history) / len(self._history)
        if avg > self.stale_threshold:
            logger.debug(
                f"[StalenessBasedPolicy] avg_staleness={avg:.3f} > {self.stale_threshold}: "
                f"decreasing C by {self.c_step}"
            )
            return current_c - self.c_step
        if avg < self.fresh_threshold:
            logger.debug(
                f"[StalenessBasedPolicy] avg_staleness={avg:.3f} < {self.fresh_threshold}: "
                f"increasing C by {self.c_step}"
            )
            return current_c + self.c_step
        return None

    def name(self) -> str:
        return "staleness_based"


class EligibleEndsBasedPolicy(DynamicKCPolicy):
    """Right-sizes C to track the actual eligible training pool.

    C is meaningful only relative to the number of trainers that can actually
    receive the training task.  If n_eligible_train << C, the selector cannot
    fill the concurrency target — those slots are wasted.  If n_eligible_train
    >> C, there is untapped parallelism.

    - ``n_eligible_train > current_c * headroom_factor``:
        pool is significantly larger than C → increase C by ``c_step``.
    - ``n_eligible_train < current_c * undercommit_factor``:
        pool is smaller than C → snap C down to the actual pool size.
    - Otherwise: no change.

    This interacts naturally with task_eligible_states: when FwdLLM expands
    the train-eligible states to include AVL_EVAL, n_eligible_train grows and
    this policy automatically increases C to exploit the larger pool.

    This policy never touches K.
    """

    def __init__(
        self,
        headroom_factor: float = 1.5,
        undercommit_factor: float = 0.8,
        c_step: int = 5,
        window: int = 5,
    ):
        self.headroom_factor = headroom_factor
        self.undercommit_factor = undercommit_factor
        self.c_step = c_step
        self._history: deque = deque(maxlen=window)

    def compute_new_k(self, current_k: int, metrics: dict) -> Optional[int]:
        return None

    def compute_new_c(self, current_c: int, metrics: dict) -> Optional[int]:
        n_eligible = metrics.get("n_eligible_train")
        if n_eligible is None:
            return None
        self._history.append(n_eligible)
        if len(self._history) < self._history.maxlen:
            return None
        avg_eligible = sum(self._history) / len(self._history)
        if avg_eligible > current_c * self.headroom_factor:
            logger.debug(
                f"[EligibleEndsBasedPolicy] avg_eligible={avg_eligible:.1f} > "
                f"C*headroom={current_c * self.headroom_factor:.1f}: increasing C by {self.c_step}"
            )
            return current_c + self.c_step
        if avg_eligible < current_c * self.undercommit_factor:
            new_c = max(1, int(avg_eligible))
            logger.debug(
                f"[EligibleEndsBasedPolicy] avg_eligible={avg_eligible:.1f} < "
                f"C*undercommit={current_c * self.undercommit_factor:.1f}: "
                f"snapping C down to {new_c}"
            )
            return new_c
        return None

    def name(self) -> str:
        return "eligible_ends_based"


class StepSchedulePolicy(DynamicKCPolicy):
    """Deterministic curriculum K decay.

    Decreases K by ``k_step`` every ``n_aggs_per_step`` aggregations, down to
    ``k_floor``.  Useful for curriculum learning where a large K in early
    rounds provides stable gradient averaging, then a smaller K in later rounds
    allows faster convergence.

    This policy never touches C.
    """

    def __init__(
        self,
        k_step: int = 2,
        n_aggs_per_step: int = 50,
        k_floor: int = 5,
    ):
        self.k_step = k_step
        self.n_aggs_per_step = n_aggs_per_step
        self.k_floor = k_floor

    def compute_new_k(self, current_k: int, metrics: dict) -> Optional[int]:
        n = metrics.get("n_aggs_completed", 0)
        if n > 0 and n % self.n_aggs_per_step == 0:
            new_k = max(self.k_floor, current_k - self.k_step)
            return new_k if new_k != current_k else None
        return None

    def compute_new_c(self, current_c: int, metrics: dict) -> Optional[int]:
        return None

    def name(self) -> str:
        return "step_schedule"


class CompositePolicy(DynamicKCPolicy):
    """Chains multiple policies.

    For K and C independently, the first sub-policy that returns a non-None
    value wins.  This allows combining orthogonal policies — e.g. a
    VarianceBasedPolicy for K and an EligibleEndsBasedPolicy for C.
    """

    def __init__(self, sub_policies: list):
        """
        Args:
            sub_policies: list of dicts, each with keys:
                "name"   (str): policy name recognised by build_policy()
                "kwargs" (dict, optional): keyword args for the policy constructor
        """
        self.policies = [
            build_policy(p["name"], p.get("kwargs", {})) for p in sub_policies
        ]

    def compute_new_k(self, current_k: int, metrics: dict) -> Optional[int]:
        for p in self.policies:
            result = p.compute_new_k(current_k, metrics)
            if result is not None:
                return result
        return None

    def compute_new_c(self, current_c: int, metrics: dict) -> Optional[int]:
        for p in self.policies:
            result = p.compute_new_c(current_c, metrics)
            if result is not None:
                return result
        return None

    def name(self) -> str:
        return f"composite[{','.join(p.name() for p in self.policies)}]"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_POLICY_REGISTRY: dict = {
    "noop": NoOpPolicy,
    "variance_based": VarianceBasedPolicy,
    "staleness_based": StalenessBasedPolicy,
    "eligible_ends_based": EligibleEndsBasedPolicy,
    "step_schedule": StepSchedulePolicy,
    "composite": CompositePolicy,
}


def build_policy(policy_name: str, policy_kwargs: dict) -> DynamicKCPolicy:
    """Instantiate a DynamicKCPolicy by name.

    Args:
        policy_name: One of the keys in _POLICY_REGISTRY.
        policy_kwargs: Keyword arguments forwarded to the policy constructor.

    Raises:
        ValueError: If policy_name is not recognised.
    """
    cls = _POLICY_REGISTRY.get(policy_name)
    if cls is None:
        raise ValueError(
            f"Unknown dynamic_kc policy: '{policy_name}'. "
            f"Valid options: {sorted(_POLICY_REGISTRY)}"
        )
    return cls(**policy_kwargs)
