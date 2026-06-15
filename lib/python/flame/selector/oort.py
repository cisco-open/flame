# Copyright 2023 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License"); you
# may not use this file except in compliance with the License. You may
# obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""OortSelector class."""

import logging
import math
import random
from datetime import timedelta
from collections import deque
import numpy as np

from flame.common.typing import Scalar
from flame.common.util import MLFramework, get_ml_framework_in_use
from flame.end import End
from flame.selector import AbstractSelector, SelectorReturnType
from flame.selector import scoring
from flame.selector.properties import (
    PROP_DATASET_SIZE,
    PROP_END_ID,
    PROP_LAST_EVAL_ROUND,
    PROP_LAST_SELECTED_ROUND,
    PROP_ROUND_DURATION,
    PROP_ROUND_START_TIME,
    PROP_SELECTED_COUNT,
    PROP_STAT_UTILITY,
    PROP_UPDATE_COUNT,
    PROP_UTILITY,
)

logger = logging.getLogger(__name__)


class OortSelector(AbstractSelector):
    """A selector class based on Oort."""

    def __init__(self, **kwargs):
        """Initailize instance."""
        super().__init__(**kwargs)

        ml_framework_in_use = get_ml_framework_in_use()
        if ml_framework_in_use != MLFramework.PYTORCH:
            raise NotImplementedError(
                "FedBalancer is currently only implemented in PyTorch;"
            )

        try:
            self.aggr_num = kwargs["aggr_num"]
        except KeyError:
            raise KeyError("aggr_num is not specified in config")

        if self.aggr_num < 0:
            self.aggr_num = 1
        self.round = 0

        # With Oort, we select 1.3 * k ends and wait until k ends to
        # complete at a round
        self.overcommitment = 1.3
        self.num_of_ends = int(self.aggr_num * self.overcommitment)

        # Algorithm hyperparameters default to the Oort paper (scoring.OORT_PAPER_DEFAULTS);
        # the refl baseline overrides a subset via selector.kwargs to match the REFL fork.
        _d = scoring.OORT_PAPER_DEFAULTS
        self.exploration_factor = kwargs.get("exploration_factor", _d["exploration_factor"])
        self.exploration_factor_decay = kwargs.get("exploration_decay", _d["exploration_decay"])
        self.min_exploration_factor = kwargs.get("exploration_min", _d["exploration_min"])

        self.exploitation_util_history = []

        # Assuming a max round duration of 99999 seconds (~1.2 days)
        self.round_preferred_duration = timedelta(seconds=99999)
        self.round_threshold = kwargs.get("round_threshold", _d["round_threshold"])
        self.pacer_delta = kwargs.get("pacer_delta", _d["pacer_delta"])
        self.pacer_step = kwargs.get("pacer_step", _d["pacer_step"])

        self.blocklist_threshold = -1

        self.alpha = kwargs.get("round_penalty", _d["round_penalty"])  # system_util exponent

        # Reference Oort normalizes+clips the reward into ~[0,1] (get_norm) before
        # adding the temporal term; the raw reward (~70) had made it inert.
        self.normalize_reward = kwargs.get("normalize_reward", True)
        self.clip_bound = kwargs.get("clip_bound", _d["clip_bound"])
        # cut_off_util: exploitation-pool breadth factor; was hardcoded 0.95.
        self.cut_off_util = kwargs.get("cut_off_util", _d["cut_off_util"])

        # Track sliding window statistics for the selector
        self._selector_stats = {}
        for task in ["train", "eval"]:
            self._selector_stats[task] = {"data": {}, "summary": {}}
            for metric in ["util", "speed", "round"]:
                for window in [50, 100, 200]:
                    key = f"{metric}_last_{window}"
                    self._selector_stats[task]["data"][key] = deque(maxlen=window)

        self._select_run_counter = 0

    def compute_trainer_stat_summary(self):
        def compute_summary(values):
            # Filter out None values
            if values is None:
                return {
                    "min": None,
                    "max": None,
                    "p25": None,
                    "p50": None,
                    "p75": None,
                }
            values = [v for v in values if v is not None]
            if not values:
                return {
                    "min": None,
                    "max": None,
                    "p25": None,
                    "p50": None,
                    "p75": None,
                }

            values = np.array(values, dtype=float)
            return {
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "p25": float(np.percentile(values, 25)),
                "p50": float(np.percentile(values, 50)),
                "p75": float(np.percentile(values, 75)),
            }

        tasks = ["train", "eval"]
        metrics = [
            "util_last_50",
            "util_last_100",
            "util_last_200",
            "speed_last_50",
            "speed_last_100",
            "speed_last_200",
            "round_last_50",
            "round_last_100",
            "round_last_200",
        ]

        for task in tasks:
            for metric in metrics:
                values = self._selector_stats[task]["data"].get(metric, [])
                key = f"stat_{metric}" if "util" in metric else metric
                self._selector_stats[task]["summary"][key] = compute_summary(values)

    def _reset_selector_stats(self) -> None:
        self._selector_stats = {}

    def select(
        self,
        ends: dict[str, End],
        channel_props: dict[str, Scalar],
        trainer_unavail_list: list,
        task_to_perform: str,
        **kwargs,
    ) -> SelectorReturnType:
        """Return k number of ends from the given ends."""
        num_of_ends = min(len(ends), self.num_of_ends)
        if num_of_ends == 0:
            logger.debug("ends is empty")
            return {}

        round = channel_props["round"] if "round" in channel_props else 0
        logger.info(
            f"let's select {num_of_ends} ends for new round {round}, task: {task_to_perform}"
        )

        # full candidate pool, captured before any filtering for telemetry
        all_ends = dict(ends)

        if round <= self.round and len(self.selected_ends) != 0:
            return {key: None for key in self.selected_ends}

        self.pacer()

        eligible_ends = {
            end_id: end
            for end_id, end in ends.items()
            if end_id not in self.selected_ends
        }

        if len(eligible_ends) == 0:
            logger.error(
                f"[OORT_SELECT] Round {round}: no eligible trainers "
                f"(total={len(ends)}, in_flight={len(self.selected_ends)})"
            )
            return {}

        if len(eligible_ends) < num_of_ends:
            logger.warning(
                f"[OORT_SELECT] Round {round}: only {len(eligible_ends)}/{num_of_ends} trainers eligible"
            )
            num_of_ends = len(eligible_ends)

        ends = eligible_ends

        # Make a filter of blocklist ends
        blocklist_end_ids = self.find_blocklists(ends)

        # Make a filter of unavailable ends
        if trainer_unavail_list != []:
            logger.debug(
                "### Oort select got non-empty trainer_unavail_list, will "
                "remove unavail trainers from round"
            )

        # get the list of unavailable_ends and pass to
        # fetch_statistical_utility treat unavailable_ends like
        # blocklist_ends inside fetch_statistical_utility

        # Make a list of tuple (end_id, end_utility) as an
        # utility_list As unexplored ends that are not selected before
        # do not have utility value, collect them separately with
        # unexplored_end_ids list
        utility_list, unexplored_end_ids = self.fetch_statistical_utility(
            ends, blocklist_end_ids, trainer_unavail_list
        )

        # This indicates the first round, where no end's utility has
        # been measured; Then, perform random selection
        if len(utility_list) == 0 and len(self.selected_ends) == 0:
            self.round = round
            result = self.select_random(ends, num_of_ends)
            self.emit_selection(
                round, task_to_perform, all_ends, ends.keys(),
                self.selected_ends, extra={"mode": "random_first_round"},
            )
            return result

        # Not the first round, performing Oort-based selection
        # Calculate number of ends to select for exploration and
        # exploitation
        (
            exploration_len,
            exploitation_len,
        ) = self.calculate_num_of_exploration_exploitation(
            num_of_ends, unexplored_end_ids
        )

        if len(utility_list) == 0:
            self.round = round
            result = self.select_random(ends, num_of_ends)
            self.emit_selection(
                round, task_to_perform, all_ends, ends.keys(),
                self.selected_ends, extra={"mode": "random_no_utility"},
            )
            return result

        utility_list = self.calculate_total_utility(utility_list, ends, round)
        cutoff_utility = self.cutoff_util(utility_list, num_of_ends)

        exploit_end_ids = self.sample_by_util(
            cutoff_utility, utility_list, exploitation_len
        )

        explore_end_ids = []
        if self.exploration_factor > 0.0 and len(unexplored_end_ids) > 0:
            explore_end_ids = self.sample_by_speed(unexplored_end_ids, exploration_len)

        newly_selected = set([*explore_end_ids, *exploit_end_ids])
        self.selected_ends = self.selected_ends | newly_selected

        self.save_exploited_utility_history(ends, exploit_end_ids)
        self.update_exploration_factor()
        self.increment_selected_count_on_selected_ends(ends)

        logger.info(f"selected ends: {self.selected_ends}")
        self.round = round

        self._select_run_counter += 1
        for selected_end_id in self.selected_ends:
            # in-flight ids may not be in the current eligible `ends`; skip them
            if selected_end_id not in ends:
                continue
            end_stat_util = ends[selected_end_id].get_property(PROP_STAT_UTILITY)
            end_speed = ends[selected_end_id].get_property(PROP_ROUND_DURATION)
            end_last_round = ends[selected_end_id].get_property(PROP_LAST_EVAL_ROUND)
            for window in [50, 100, 200]:
                if end_stat_util is not None:
                    self._selector_stats[task_to_perform]["data"][
                        f"util_last_{window}"
                    ].append(end_stat_util)
                if end_speed is not None:
                    self._selector_stats[task_to_perform]["data"][
                        f"speed_last_{window}"
                    ].append(end_speed.total_seconds())
                if end_last_round is not None:
                    self._selector_stats[task_to_perform]["data"][
                        f"round_last_{window}"
                    ].append(end_last_round)

        if self._select_run_counter % 5 == 0:
            self.compute_trainer_stat_summary()
            logger.info(
                f"Train selector stats summary: {self._selector_stats['train']['summary']}"
            )
            logger.info(
                f"Eval selector stats summary: {self._selector_stats['eval']['summary']}"
            )
            self._select_run_counter = 0

        # system_util = (pref/round_duration)^alpha depends on the dynamic
        # round_preferred_duration (a per-round percentile of candidate durations);
        # emit it so a divergence can be traced to the target vs the duration input.
        _pref = getattr(self, "round_preferred_duration", None)
        self.emit_selection(
            round, task_to_perform, all_ends, eligible_ends.keys(),
            self.selected_ends,
            per_trainer_extra=getattr(self, "_audit_components", None),
            extra={
                "exploration_factor": self.exploration_factor,
                "explore_ids": list(explore_end_ids),
                "exploit_ids": list(exploit_end_ids),
                "round_preferred_duration_s": _pref.total_seconds()
                if hasattr(_pref, "total_seconds") else _pref,
                "alpha": getattr(self, "alpha", None),
                # per-round speed-penalty summary over selected (see _system_util_summary)
                **self._system_util_summary(),
            },
        )
        return {key: None for key in self.selected_ends}

    def cutoff_util(
        self,
        sorted_utility_list: list[tuple[str, float]],
        num_of_ends: int,
    ) -> float:
        """Cutoff utility = cut_off_util * the (exploitLen-th HIGHEST) score.

        Reference Oort thresholds at the exploitation boundary's score then samples
        above it (oort.py:329). `sorted_utility_list` is ASCENDING, so the
        exploitLen-th highest is at index ``len-1-exploitLen``. The prior port
        indexed near the bottom, making the factor inert.
        """
        if not sorted_utility_list:
            logger.debug("Got empty utility_list, returning 999999.0")
            return 999999.0

        exploit_len = int(num_of_ends * (1.0 - self.exploration_factor))
        index = len(sorted_utility_list) - 1 - exploit_len
        index = max(0, min(index, len(sorted_utility_list) - 1))

        return self.cut_off_util * sorted_utility_list[index][PROP_UTILITY]

    def sample_by_util(
        self,
        cutoff_utility: float,
        utility_list: list[dict[str, Scalar]],
        num_of_ends: int,
    ) -> list[str]:
        """Sample num_of_ends clients by utility."""

        over_cutoff_utility_end_ids = []
        over_cutoff_utility_probs = []
        over_cutoff_utility_sum = 0

        under_cutoff_utility_list = []

        # Divide ends on whether its utility exceeds cutoff_loss or
        # not
        for utility_pair in utility_list:
            if utility_pair[PROP_UTILITY] >= cutoff_utility:
                over_cutoff_utility_end_ids.append(utility_pair[PROP_END_ID])
                over_cutoff_utility_probs.append(utility_pair[PROP_UTILITY])
                over_cutoff_utility_sum += utility_pair[PROP_UTILITY]
            else:
                under_cutoff_utility_list.append(utility_pair)

        # Select clients on the probability based on the utility
        # divided by the utility sum
        for prob_idx in range(len(over_cutoff_utility_probs)):
            over_cutoff_utility_probs[prob_idx] /= over_cutoff_utility_sum

        selected_ends = self._rng.choice(
            over_cutoff_utility_end_ids,
            size=min(len(over_cutoff_utility_end_ids), num_of_ends),
            replace=False,
            p=over_cutoff_utility_probs,
        )

        # np.random.choice yields np.str_ entries; cast to plain str so the
        # ids match the python-str keys of the ``ends`` dict downstream.
        return [str(e) for e in selected_ends]

    def sample_by_speed(
        self, unexplored_end_ids: list[str], num_of_ends: int
    ) -> list[str]:
        """Sample num_of_ends clients by speed."""

        # Oort paper prioritizes unexplored ends with faster system
        # speed We initially implement to perform random here
        # Cast np.str_ -> str so ids match the python-str keys of ``ends``.
        return [
            str(e)
            for e in self._rng.choice(
                unexplored_end_ids, size=num_of_ends, replace=False
            )
        ]

    def pacer(self) -> None:
        """
        Controls round preferred duration based on the exploited
        statistical utility.
        """

        if (
            len(self.exploitation_util_history) >= 2 * self.pacer_step
            and self.round % self.pacer_step == 0
        ):
            last_pacer_step_util = sum(
                self.exploitation_util_history[-2 * self.pacer_step : -self.pacer_step]
            )
            curr_pacer_step_util = sum(
                self.exploitation_util_history[-self.pacer_step :]
            )
            if last_pacer_step_util > curr_pacer_step_util:
                self.round_threshold = min(
                    100.0, self.round_threshold + self.pacer_delta
                )

    def find_blocklists(self, ends: dict[str, End]) -> list[str]:
        """Make a filter of blocklist ends."""

        blocklist_end_ids = []
        if self.blocklist_threshold != -1:
            for end_id in ends.keys():
                if (
                    ends[end_id].get_property(PROP_SELECTED_COUNT)
                    > self.blocklist_threshold
                ):
                    blocklist_end_ids.append(end_id)
        return blocklist_end_ids

    def calculate_num_of_exploration_exploitation(
        self, num_of_ends: int, unexplored_end_ids: list[str]
    ) -> tuple[int, int]:
        """Split num_of_ends into (exploration, exploitation) counts."""
        exploration_len = min(
            int(num_of_ends * self.exploration_factor) + 1,
            len(unexplored_end_ids),
        )
        return exploration_len, num_of_ends - exploration_len

    def fetch_statistical_utility(
        self,
        ends: dict[str, End],
        blocklist_end_ids: list[str],
        trainer_unavail_list: list[str],
    ) -> tuple[list[tuple[str, float]], list[str]]:
        """Return (utility_list, unexplored_end_ids)."""
        utility_list = []
        unexplored_end_ids = []

        for end_id in ends.keys():
            if (end_id not in blocklist_end_ids) and (
                end_id not in trainer_unavail_list
            ):
                end_utility = ends[end_id].get_property(PROP_STAT_UTILITY)
                if end_utility is not None:
                    utility_list.append(
                        {PROP_END_ID: end_id, PROP_UTILITY: end_utility}
                    )
                else:
                    unexplored_end_ids.append(end_id)

        return utility_list, unexplored_end_ids

    def calculate_round_preferred_duration(self, ends: dict[str, End]) -> float:
        """Preferred round duration based on round_threshold + observed end durations."""
        if self.round_threshold < 100.0:
            sorted_round_duration = []
            for end_id in ends.keys():
                end_round_duration = ends[end_id].get_property(PROP_ROUND_DURATION)
                if end_round_duration is not None:
                    sorted_round_duration.append(end_round_duration)
                else:
                    sorted_round_duration.append(timedelta(seconds=60))
            # pref = round_threshold-th PERCENTILE -> sort first (ref Oort oort.py:272)
            sorted_round_duration.sort()
            round_preferred_duration = timedelta(
                seconds=sorted_round_duration[
                    min(
                        int(len(sorted_round_duration) * self.round_threshold / 100.0),
                        len(sorted_round_duration) - 1,
                    )
                ].total_seconds()
            )
        else:
            # Assuming a max round duration of 99999 seconds (~1.2
            # days)
            round_preferred_duration = timedelta(seconds=99999)

        logger.debug(f"returning round_preferred_duration: {round_preferred_duration}")
        return round_preferred_duration

    def _system_util_summary(self) -> dict:
        """Per-round speed-penalty summary over selected ends, for telemetry.

        When `pref` is non-binding the system_util penalty never fires and the
        selector ignores speed; logging this makes that visible without recompute.
        """
        audit = getattr(self, "_audit_components", None) or {}
        sel = [
            audit[e]["system_util"]
            for e in getattr(self, "selected_ends", [])
            if e in audit and audit[e].get("system_util") is not None
        ]
        if not sel:
            return {"sys_util_mean": None, "frac_penalized": None, "pref_binds": None}
        penalized = sum(1 for su in sel if su < 1.0)
        return {
            "sys_util_mean": sum(sel) / len(sel),
            "frac_penalized": penalized / len(sel),
            "pref_binds": penalized > 0,
        }

    def calculate_temporal_uncertainty_of_trainer(
        self, ends: dict[str, End], end_id: str, round: int
    ) -> float:
        """
        Calculate temproal uncertainty term based on the end's last
        selected round.
        """

        # NOTE: reference Oort keys this on the round the util was last UPDATED (on
        # completion), not last SELECTED — would need a new aggregator-stamped
        # property across real+sim. Subtle effect; deferred.
        end_last_selected_round = ends[end_id].get_property(PROP_LAST_SELECTED_ROUND)
        return scoring.oort_temporal_uncertainty(round, end_last_selected_round)

    def calculate_global_system_utility_of_trainer(
        self, ends: dict[str, End], end_id: str
    ) -> float:
        """
        Calculate global system utility based on the end's round
        duration.
        """

        end_round_duration = ends[end_id].get_property(PROP_ROUND_DURATION)

        if end_round_duration is None:
            return 1
        pref = self.round_preferred_duration
        return scoring.oort_system_utility(
            end_round_duration.total_seconds(),
            pref.total_seconds() if pref is not None else None,
            self.alpha,
        )

    def save_exploited_utility_history(
        self, ends: dict[str, End], exploit_end_ids: list[str]
    ) -> None:
        # exploit_end_ids may be a numpy array (from sample_by_util); use len()
        # so the emptiness check doesn't raise "truth value ambiguous".
        if len(exploit_end_ids) == 0:
            return
        total = sum(
            ends[eid].get_property(PROP_STAT_UTILITY) for eid in exploit_end_ids
        )
        self.exploitation_util_history.append(total / len(exploit_end_ids))

    def update_exploration_factor(self) -> None:
        self.exploration_factor = max(
            self.exploration_factor * self.exploration_factor_decay,
            self.min_exploration_factor,
        )

    def increment_selected_count_on_selected_ends(self, ends: dict[str, End]) -> None:
        for end_id in self.selected_ends:
            # selected_ends may hold in-flight ids no longer in the current
            # eligible `ends`; skip those rather than KeyError.
            if end_id not in ends:
                continue
            count = ends[end_id].get_property(PROP_SELECTED_COUNT) or 0
            ends[end_id].set_property(PROP_SELECTED_COUNT, count + 1)

    def select_random(self, ends: dict[str, End], num_of_ends: int) -> dict[str, None]:
        """Randomly select num_of_ends ends, merging with any in-flight set."""
        newly_selected = set(self._pyrng.sample(sorted(ends), num_of_ends))
        self.selected_ends = self.selected_ends | newly_selected
        return {key: None for key in newly_selected}

    def calculate_total_utility(
        self, utility_list: list[tuple[str, float]], ends: dict[str, End], round: int
    ) -> list[tuple[str, float]]:
        """Apply temporal uncertainty and global system utility to each entry."""
        self.round_preferred_duration = self.calculate_round_preferred_duration(ends)

        utility_list = sorted(utility_list, key=lambda x: x[PROP_UTILITY])

        # Normalize+clip the statistical reward across this round's candidates
        # (reference Oort get_norm) so `believed_I` lands in ~[0,1] and the additive
        # temporal/UCB term is meaningful. Stats computed once over the raw rewards.
        if self.normalize_reward and utility_list:
            _min, _range, _clip = scoring.oort_norm_stats(
                [u[PROP_UTILITY] for u in utility_list], self.clip_bound
            )
        else:
            _min, _range, _clip = None, None, None

        # Per-candidate score components stashed for the offline staleness audit
        # (believed I_m, temporal, system_util); read in emit_selection. Reset
        # per round so REFL's multiple per-group calls accumulate within a round.
        if getattr(self, "_audit_round", None) != round:
            self._audit_components = {}
            self._audit_round = round
        for utility_idx in range(len(utility_list)):
            stat_util = utility_list[utility_idx][PROP_UTILITY]
            if self.normalize_reward and _range is not None:
                stat_util = scoring.oort_normalize_reward(
                    stat_util, _min, _range, _clip
                )
            curr_end_id = utility_list[utility_idx][PROP_END_ID]

            # Score = (stat_util + temporal) * system_util, via the shared pure
            # scorer so the offline staleness audit reproduces it exactly.
            temporal = self.calculate_temporal_uncertainty_of_trainer(
                ends, curr_end_id, round
            )
            system_util = self.calculate_global_system_utility_of_trainer(
                ends, curr_end_id
            )
            self._audit_components[curr_end_id] = {
                "believed_I": stat_util,
                "temporal": temporal,
                "system_util": system_util,
            }
            utility_list[utility_idx][PROP_UTILITY] = scoring.oort_combine_score(
                stat_util, temporal, system_util
            )

        utility_list = sorted(utility_list, key=lambda x: x[PROP_UTILITY])

        return utility_list

    def _cleanup_removed_ends(self, end_id):
        logger.debug(f"end_id {end_id} left the channel")

    def _cleanup_recvd_ends(self, ends: dict[str, End]):
        """Free ends whose updates were received from the in-flight set."""
        if not self.ordered_updates_recv_ends:
            return
        for end_id in self.ordered_updates_recv_ends:
            self.selected_ends.discard(end_id)
        logger.debug(
            f"freed {len(self.ordered_updates_recv_ends)} ends; "
            f"in-flight now {len(self.selected_ends)}"
        )
        self.ordered_updates_recv_ends = []
