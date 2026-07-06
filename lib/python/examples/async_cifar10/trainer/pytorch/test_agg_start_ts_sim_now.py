# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Batch 3 T3.0: PyTorchCifar10Trainer._sim_now()'s real-mode branch.

Real mode anchors to `_agg_start_origin` (broadcast by the aggregator and
cached by flame.mode.horizontal.syncfl.trainer.Trainer -- see
lib/python/tests/mode/test_agg_start_ts_broadcast.py for that half) rather
than this trainer's own `trainer_start_ts`, so every trainer's trace lookups
share the exact origin the aggregator uses. See UNAVAILABILITY_DESIGN.md
Batch 3 T3.0.
"""

import time

import pytest

from trainer.pytorch.main import PyTorchCifar10Trainer


def _make(*, simulated, sim_send_ts=None, agg_start_origin=None, trainer_start_ts=None):
    t = PyTorchCifar10Trainer.__new__(PyTorchCifar10Trainer)
    t.simulated = simulated
    t._sim_send_ts = sim_send_ts
    t._agg_start_origin = agg_start_origin
    t.trainer_start_ts = trainer_start_ts if trainer_start_ts is not None else time.time()
    return t


class TestSimNowRealModeUsesSharedOrigin:
    def test_uses_broadcast_origin_when_available(self):
        now = time.time()
        t = _make(
            simulated=False, agg_start_origin=now - 50.0, trainer_start_ts=now - 999.0
        )
        assert t._sim_now() == pytest.approx(50.0, abs=0.5)

    def test_falls_back_to_trainer_start_ts_before_first_dispatch(self):
        now = time.time()
        t = _make(simulated=False, agg_start_origin=None, trainer_start_ts=now - 12.0)
        assert t._sim_now() == pytest.approx(12.0, abs=0.5)

    def test_sim_mode_unaffected_by_origin_fields(self):
        t = _make(simulated=True, sim_send_ts=88.0, agg_start_origin=12345.0)
        assert t._sim_now() == 88.0
