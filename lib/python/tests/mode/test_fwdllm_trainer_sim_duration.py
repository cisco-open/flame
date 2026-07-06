# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""fwdllm's trainer (FedSgdTrainer.py) reported only real_gpu_time_s, not
sim_round_duration_s -- unlike async_cifar10's trainer, which reports total
round wall time (gpu + modeled delay). fwdllm has no budget-vs-actual
contention model (its delay is a flat additive sleep, not a sleep-to-fill-
budget pattern), so only sim_round_duration_s is added here -- NOT
training_budget_s/overran/remaining_time_s, which would need a budget
concept fwdllm doesn't have (see ../../examples/MIGRATING_TO_LAUNCHER.md §9).

This covers _emulate_training_delay()'s return-value change: it now returns
the seconds actually slept (0.0 if delay emulation is disabled), which the
caller adds to real_gpu_time_s to report sim_round_duration_s.
"""

import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "examples", "fwdllm",
        "trainer", "forward_training",
    ),
)

from FedSgdTrainer import FedSGDTrainer  # noqa: E402


class _FakeTrainer:
    """Minimal stand-in exposing only the state _emulate_training_delay
    touches; binds the real method under test."""

    _emulate_training_delay = FedSGDTrainer._emulate_training_delay

    def __init__(self, training_delay_enabled, training_delay_s=0.0,
                 training_delay_factor=1.0, speedup_factor=1.0):
        self.training_delay_enabled = training_delay_enabled
        self.training_delay_s = training_delay_s
        self.training_delay_factor = training_delay_factor
        self.speedup_factor = speedup_factor
        self.trainer_id = "t1"


class TestEmulateTrainingDelayReturnsSleptSeconds:
    def test_returns_zero_when_disabled(self):
        t = _FakeTrainer(training_delay_enabled="False", training_delay_s=10.0)
        assert t._emulate_training_delay() == 0.0

    def test_returns_computed_delay_when_enabled(self):
        t = _FakeTrainer(
            training_delay_enabled="True", training_delay_s=3.0,
            training_delay_factor=1.0, speedup_factor=1.0,
        )
        assert t._emulate_training_delay() == 3.0

    def test_speedup_factor_scales_the_returned_delay(self):
        """The returned value must match what was actually slept (eval_delay
        / speedup_factor), not the unscaled eval_delay -- otherwise
        sim_round_duration_s would overstate the real wall time under a
        speedup."""
        t = _FakeTrainer(
            training_delay_enabled="True", training_delay_s=10.0,
            training_delay_factor=2.0, speedup_factor=5.0,
        )
        # eval_delay = 10.0 / 2.0 = 5.0; slept = 5.0 / 5.0 = 1.0
        assert t._emulate_training_delay() == 1.0
