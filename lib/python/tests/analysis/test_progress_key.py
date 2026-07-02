# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for analyze_run.py's progress_key() round-collapse fix.

fwdllm-family runs can sit at trainer_round.round == 1 for hundreds of events
(a round only completes once all data bins finish), which collapsed any
analyzer plot bucketing by plain `round` onto a single x-value. progress_key()
folds in `data_id` (present on trainer_round records only) to restore a real
ordering, while staying a no-op for async_cifar10 (no `data_id` field).
"""

import os
import sys

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..",
                 "scripts", "analysis"),
)

from analyze_run import progress_key, trainer_rounds_by_round  # noqa: E402


def test_no_data_id_falls_back_to_plain_round():
    """async_cifar10 trainer_round records carry no data_id -- no-op."""
    assert progress_key({"round": 7}) == 7
    assert progress_key({"round": 0}) == 0


def test_data_id_folds_into_round_major_ordering():
    """fwdllm trainer_round records: round is major, data_id is minor."""
    assert progress_key({"round": 1, "data_id": 0}) == 200
    assert progress_key({"round": 1, "data_id": 50}) == 250
    assert progress_key({"round": 2, "data_id": 0}) == 400


def test_data_id_ordering_never_collides_across_rounds():
    """A later round's earliest data_id must sort after an earlier round's
    latest data_id -- this is only guaranteed because the multiplier (200)
    exceeds fwdllm's total_data_bins (150)."""
    end_of_round_1 = progress_key({"round": 1, "data_id": 149})
    start_of_round_2 = progress_key({"round": 2, "data_id": 0})
    assert end_of_round_1 < start_of_round_2


def test_fwdllm_round_collapse_is_fixed():
    """The concrete bug: 733 events all at round==1 collapse to one bucket
    under plain round, but spread out under progress_key because data_id
    varies 0..50 across them."""
    records = [
        {"event": "trainer_round", "round": 1, "data_id": data_id}
        for data_id in range(51)
        for _ in range(14)  # ~733 events total, matching the real run's scale
    ]
    plain_rounds = {r["round"] for r in records}
    assert len(plain_rounds) == 1  # the collapse, reproduced

    trbr = trainer_rounds_by_round(records)
    assert len(trbr) == 51  # progress_key fixes it


def test_trainer_rounds_by_round_ignores_non_trainer_round_events():
    records = [
        {"event": "trainer_round", "round": 1, "data_id": 0},
        {"event": "selection", "round": 1},
    ]
    trbr = trainer_rounds_by_round(records)
    assert list(trbr.keys()) == [200]
    assert len(trbr[200]) == 1
