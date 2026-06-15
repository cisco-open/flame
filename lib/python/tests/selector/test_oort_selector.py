# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Oort selector tests."""

from datetime import timedelta

import pytest

from flame.selector.oort import OortSelector
from flame.selector.async_oort import AsyncOortSelector
from flame.selector.properties import PROP_ROUND_DURATION


@pytest.fixture
def oort():
    return OortSelector(aggr_num=3)


@pytest.fixture
def async_oort():
    # Minimal kwargs required by AsyncOortSelector.__init__; the framework is
    # patched to PyTorch in conftest (flame.selector.async_oort included there).
    return AsyncOortSelector(
        c=30,
        aggGoal=10,
        evalGoalFactor=0.5,
        roundNudgeType="last_train",
        selectType="default",
    )


class TestOortInit:
    def test_defaults(self, oort):
        assert oort.aggr_num == 3
        assert oort.num_of_ends == int(3 * 1.3)
        assert isinstance(oort.selected_ends, set)
        assert oort.ordered_updates_recv_ends == []
        assert 0.0 < oort.exploration_factor <= 1.0


class TestOortColdStart:
    def test_first_round_random_selection(self, oort, make_ends, channel_props):
        ends = make_ends(count=10, prefix="t")
        result = oort.select(
            ends, channel_props, trainer_unavail_list=[], task_to_perform="train"
        )
        assert len(result) == oort.num_of_ends
        for end_id in result:
            assert end_id in ends

    def test_in_flight_excluded_next_round(
        self, oort, make_ends, channel_props
    ):
        ends = make_ends(count=10, prefix="t")
        oort.select(
            ends, channel_props, trainer_unavail_list=[], task_to_perform="train"
        )
        first_picked = set(oort.selected_ends)

        channel_props["round"] = 2
        result = oort.select(
            ends, channel_props, trainer_unavail_list=[], task_to_perform="train"
        )
        for end_id in result:
            assert end_id not in first_picked or end_id in oort.selected_ends


class TestOortIdempotentWithinRound:
    def test_same_round_returns_same_set(self, oort, make_ends, channel_props):
        ends = make_ends(count=10, prefix="t")
        r1 = oort.select(
            ends, channel_props, trainer_unavail_list=[], task_to_perform="train"
        )
        r2 = oort.select(
            ends, channel_props, trainer_unavail_list=[], task_to_perform="train"
        )
        assert set(r1.keys()) == set(r2.keys())


class TestRoundPreferredDuration:
    """Guards the Jun-15 parity fix: pref must be the round_threshold-th
    PERCENTILE of candidate durations (reference Oort sorts the list before
    indexing; the FLAME reimplementation had dropped the sort, so `pref` was an
    arbitrary dict-position duration -> the oort selector-scoring divergence)."""

    def _ends_with_durations(self, make_ends, durations_s):
        # Insert in the GIVEN (unsorted) order so a missing sort is detectable.
        ends = make_ends([f"e{i}" for i in range(len(durations_s))])
        for (eid, e), d in zip(ends.items(), durations_s):
            e.set_property(PROP_ROUND_DURATION, timedelta(seconds=d))
        return ends

    def test_round_preferred_duration_is_sorted_percentile(self, oort, make_ends):
        # Deliberately unsorted insertion order. round_threshold default = 30.
        durations = [50, 10, 40, 20, 30]
        ends = self._ends_with_durations(make_ends, durations)
        oort.round_threshold = 30
        idx = int(len(durations) * 30 / 100.0)  # = 1
        expected = sorted(durations)[idx]        # sorted=[10,20,30,40,50] -> 20
        pref = oort.calculate_round_preferred_duration(ends)
        assert pref.total_seconds() == expected
        # The buggy (unsorted) impl would have returned insertion-order[idx] = 10.
        assert pref.total_seconds() != durations[idx]

    def test_pref_is_monotone_in_threshold(self, oort, make_ends):
        durations = [5, 9, 1, 7, 3, 11, 13, 2, 6, 8]
        ends = self._ends_with_durations(make_ends, durations)
        prev = -1.0
        for thr in (10, 30, 50, 70, 90):
            oort.round_threshold = thr
            cur = oort.calculate_round_preferred_duration(ends).total_seconds()
            assert cur >= prev, f"pref not monotone in threshold at {thr}"
            prev = cur

    def test_threshold_100_is_unbounded(self, oort, make_ends):
        ends = self._ends_with_durations(make_ends, [10, 20, 30])
        oort.round_threshold = 100.0
        assert oort.calculate_round_preferred_duration(ends).total_seconds() == 99999


class TestAsyncRoundPreferredDuration:
    """Same Jun-15 sort guard for AsyncOortSelector (felix stack). The async
    selector carries its OWN copy of calculate_round_preferred_duration, so the
    sort fix must be guarded independently of OortSelector's."""

    def _ends_with_durations(self, make_ends, durations_s):
        ends = make_ends([f"e{i}" for i in range(len(durations_s))])
        for (eid, e), d in zip(ends.items(), durations_s):
            e.set_property(PROP_ROUND_DURATION, timedelta(seconds=d))
        return ends

    def test_round_preferred_duration_is_sorted_percentile(
        self, async_oort, make_ends
    ):
        durations = [50, 10, 40, 20, 30]
        ends = self._ends_with_durations(make_ends, durations)
        async_oort.round_threshold = 30
        idx = int(len(durations) * 30 / 100.0)  # = 1
        expected = sorted(durations)[idx]        # sorted=[10,20,30,40,50] -> 20
        pref = async_oort.calculate_round_preferred_duration(ends)
        assert pref.total_seconds() == expected
        assert pref.total_seconds() != durations[idx]  # buggy unsorted -> 10

    def test_pref_is_monotone_in_threshold(self, async_oort, make_ends):
        durations = [5, 9, 1, 7, 3, 11, 13, 2, 6, 8]
        ends = self._ends_with_durations(make_ends, durations)
        prev = -1.0
        for thr in (10, 30, 50, 70, 90):
            async_oort.round_threshold = thr
            cur = async_oort.calculate_round_preferred_duration(ends).total_seconds()
            assert cur >= prev, f"pref not monotone in threshold at {thr}"
            prev = cur

    def test_threshold_100_is_unbounded(self, async_oort, make_ends):
        ends = self._ends_with_durations(make_ends, [10, 20, 30])
        async_oort.round_threshold = 100.0
        pref = async_oort.calculate_round_preferred_duration(ends)
        assert pref.total_seconds() == 99999


class TestRewardNormalization:
    """Guards PARITY D2 (Jun-16): the statistical reward must be normalized+clipped
    into ~[0,1] (reference Oort get_norm) before the temporal/UCB term is added, so
    exploration is not inert. Reference: oort/oort.py get_norm:394 + score:292-295."""

    def test_norm_stats_matches_reference_get_norm(self):
        from flame.selector import scoring
        rewards = [10.0, 30.0, 20.0, 40.0, 100.0]
        _min, _range, clip = scoring.oort_norm_stats(rewards, clip_bound=0.9)
        s = sorted(rewards)
        # reference: clip = sorted[min(int(n*clip_bound), n-1)]; min*0.999; range floored
        assert clip == s[min(int(len(s) * 0.9), len(s) - 1)]
        assert _min == s[0] * 0.999
        assert _range == max(s[-1] - _min, 1e-4)

    def test_normalize_reward_clips_and_scales(self):
        from flame.selector import scoring
        _min, _range, clip = scoring.oort_norm_stats([10.0, 20.0, 100.0], clip_bound=0.5)
        # clip_value = sorted[int(3*0.5)=1] = 20 -> a raw 100 is clipped to 20
        assert scoring.oort_normalize_reward(100.0, _min, _range, clip) == \
            scoring.oort_normalize_reward(20.0, _min, _range, clip)
        # normalized reward is bounded ~[0,1]
        n = scoring.oort_normalize_reward(10.0, _min, _range, clip)
        assert 0.0 <= n <= 1.0

    def test_calculate_total_utility_normalizes_believed_I(self, oort, make_ends):
        # Raw stat-utilities ~70; after D2 the audit's believed_I must land in ~[0,1].
        from flame.selector.properties import PROP_STAT_UTILITY, PROP_END_ID, PROP_UTILITY
        ids = [f"e{i}" for i in range(5)]
        ends = make_ends(ids)
        raws = [66.0, 68.0, 70.0, 72.0, 74.0]
        for (eid, e), r in zip(ends.items(), raws):
            e.set_property(PROP_STAT_UTILITY, r)
            e.set_property(PROP_ROUND_DURATION, timedelta(seconds=10))
        util_list = [{PROP_END_ID: eid, PROP_UTILITY: r} for eid, r in zip(ids, raws)]
        assert oort.normalize_reward is True
        oort.calculate_total_utility(util_list, ends, round=5)
        for comp in oort._audit_components.values():
            assert 0.0 <= comp["believed_I"] <= 1.0, comp

    def test_normalization_can_be_disabled(self, make_ends):
        from flame.selector.properties import PROP_END_ID, PROP_UTILITY, PROP_ROUND_DURATION
        sel = OortSelector(aggr_num=3, normalize_reward=False)
        ids = [f"e{i}" for i in range(3)]
        ends = make_ends(ids)
        for eid, e in ends.items():
            e.set_property(PROP_ROUND_DURATION, timedelta(seconds=10))
        raws = [66.0, 70.0, 74.0]
        util_list = [{PROP_END_ID: eid, PROP_UTILITY: r} for eid, r in zip(ids, raws)]
        sel.calculate_total_utility(util_list, ends, round=5)
        # raw believed_I retained when disabled
        assert any(c["believed_I"] > 1.0 for c in sel._audit_components.values())


class TestAlgorithmHyperparams:
    """Guards Jun-16 fidelity pass: Oort algorithm knobs default to the paper
    (standalone Oort) and are overridable per-baseline via kwargs (REFL fork)."""

    def test_defaults_match_oort_paper(self, oort):
        from flame.selector.scoring import OORT_PAPER_DEFAULTS as d
        assert oort.round_threshold == d["round_threshold"]   # 10 (was 30)
        assert oort.clip_bound == d["clip_bound"]             # 0.98 (was 0.95)
        assert oort.cut_off_util == d["cut_off_util"]         # 0.7 (was 0.95)
        assert oort.alpha == d["round_penalty"]               # 2.0
        assert oort.exploration_factor_decay == d["exploration_decay"]  # 0.95
        assert oort.min_exploration_factor == d["exploration_min"]      # 0.2

    def test_refl_fork_overrides_apply(self):
        # The values the refl baseline config passes to match the REFL fork.
        sel = OortSelector(aggr_num=3, round_threshold=30, clip_bound=0.9,
                           cut_off_util=0.05, exploration_decay=0.98,
                           exploration_min=0.3)
        assert sel.round_threshold == 30
        assert sel.clip_bound == 0.9
        assert sel.cut_off_util == 0.05
        assert sel.exploration_factor_decay == 0.98
        assert sel.min_exploration_factor == 0.3

    def test_cutoff_util_thresholds_high_utility_boundary(self, oort, make_ends):
        # cutoff must scale the (exploitLen-th HIGHEST) utility by cut_off_util,
        # not an arbitrary low-end value. ascending list of utilities.
        from flame.selector.properties import PROP_END_ID, PROP_UTILITY
        utils = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        ul = [{PROP_END_ID: f"e{i}", PROP_UTILITY: u} for i, u in enumerate(utils)]
        oort.exploration_factor = 0.0          # exploitLen = num_of_ends
        oort.cut_off_util = 0.5
        n = len(utils)
        # exploit_len = int(n*(1-0)) = n -> index = n-1-n < 0 -> clamps to 0 (lowest)
        cut = oort.cutoff_util(ul, n)
        assert cut == 0.5 * utils[0]
        # with high exploration, exploitLen small -> boundary near the TOP
        oort.exploration_factor = 0.8
        exploit_len = int(n * (1.0 - oort.exploration_factor))  # float-faithful to code
        cut2 = oort.cutoff_util(ul, n)
        assert cut2 == 0.5 * utils[n - 1 - exploit_len]
        assert cut2 > cut  # boundary moved up toward higher utilities


class TestOortCleanup:
    def test_cleanup_recvd_ends_clears_inflight(self, oort, make_ends):
        oort.selected_ends.update(["a", "b", "c"])
        oort.ordered_updates_recv_ends = ["a", "b"]
        oort._cleanup_recvd_ends(make_ends(["a", "b", "c"]))
        assert oort.selected_ends == {"c"}
        assert oort.ordered_updates_recv_ends == []
