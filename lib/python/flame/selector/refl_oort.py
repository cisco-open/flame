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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""REFL-enhanced OortSelector with priority-based selection and pacer."""

import logging
import random
from typing import Dict, List, Set, Optional
from collections import deque
import numpy as np

from flame.common.typing import Scalar
from flame.common.util import MLFramework, get_ml_framework_in_use
from flame.end import End
from flame.selector.oort import OortSelector
from flame.selector.properties import (
    PROP_END_ID,
    PROP_SELECTED_COUNT,
    PROP_UTILITY,
)
from flame.availability.refl_tracker import REFLAvailabilityTracker

logger = logging.getLogger(__name__)


class REFLOortSelector(OortSelector):
    """
    REFL-enhanced Oort selector with availability-aware priority selection.

    Extends base OortSelector with:
    - Priority-based client selection using availability predictions
    - Adaptive pacer mechanism for round threshold adjustment
    - Blacklisting to prevent over-selection of specific clients
    """

    def __init__(self, **kwargs):
        """Initialize REFL Oort selector."""
        super().__init__(**kwargs)

        ml_framework_in_use = get_ml_framework_in_use()
        if ml_framework_in_use != MLFramework.PYTORCH:
            raise NotImplementedError(
                "REFLOortSelector is currently only implemented in PyTorch"
            )

        # REFL-specific parameters
        self.avail_priority = kwargs.get(
            "avail_priority", 0
        )  # 0=none, 1=fill, 2=strict
        self.avail_probability = kwargs.get("avail_probability", 1.0)  # Accuracy 0-1

        # Blacklisting parameters
        self.blacklist_rounds = kwargs.get("blacklist_rounds", -1)  # -1 disables
        self.blacklist_max_len = kwargs.get("blacklist_max_len", 0.3)  # Max 30%

        # Pacer parameters
        self.pacer_step = kwargs.get("pacer_step", 20)
        self.pacer_delta = kwargs.get("pacer_delta", 5)

        # Availability tracker
        trace_file = kwargs.get("availability_trace_file", None)
        trainer_registry = kwargs.get("trainer_registry_file", None)

        if trace_file:
            self.avail_tracker = REFLAvailabilityTracker(trace_file, trainer_registry)
        else:
            self.avail_tracker = None
            if self.avail_priority > 0:
                logger.warning(
                    "avail_priority > 0 but no availability_trace_file provided. "
                    "Priority selection will be disabled."
                )
                self.avail_priority = 0

        self.exploitation_util_history = deque(maxlen=200)
        self.last_pacer_round = 0
        self.newly_selected_this_round: set = set()

        logger.info(
            f"REFLOortSelector initialized: "
            f"avail_priority={self.avail_priority}, "
            f"blacklist_rounds={self.blacklist_rounds}, "
            f"pacer_step={self.pacer_step}"
        )

    def select(
        self,
        ends: Dict[str, End],
        channel_props: Dict[str, Scalar],
        trainer_unavail_list: List,
        task_to_perform: str,
        **kwargs,
    ):
        """
        Select clients using REFL's priority-based approach for SyncFL.

        Args:
            ends: Dictionary of available ends
            channel_props: Channel properties including round number
            trainer_unavail_list: List of unavailable trainers
            task_to_perform: Task type (train/eval)
            **kwargs: Additional arguments (e.g., round_duration for priority calc)

        Returns:
            Dictionary of selected end_ids
        """
        logger.debug("Calling REFL Oort select")

        num_of_ends = min(len(ends), self.num_of_ends)
        if num_of_ends == 0:
            logger.debug("ends is empty")
            return {}

        round_num = channel_props.get("round", 0)
        cur_time = channel_props.get("cur_time", 0)
        round_duration_hint = kwargs.get("round_duration", 100.0)

        logger.info(
            f"REFL Oort selecting {num_of_ends} ends for round {round_num}, "
            f"task: {task_to_perform}, avail_priority={self.avail_priority}"
        )

        if round_num <= self.round and len(self.newly_selected_this_round) != 0:
            return {key: None for key in self.newly_selected_this_round}

        self.pacer()

        unavail_set = set(trainer_unavail_list) if trainer_unavail_list else set()

        eligible_ends = {
            end_id: end
            for end_id, end in ends.items()
            if end_id not in unavail_set and end_id not in self.selected_ends
        }

        if len(eligible_ends) == 0:
            logger.error(
                f"[REFL_SELECT] Round {round_num}: no eligible trainers "
                f"(total={len(ends)}, unavail={len(unavail_set)}, in_flight={len(self.selected_ends)})"
            )
            return {}

        num_to_select = min(num_of_ends, len(eligible_ends))
        if num_to_select < num_of_ends:
            logger.warning(
                f"[REFL_SELECT] Round {round_num}: only {num_to_select}/{num_of_ends} trainers eligible"
            )

        # Build blacklist if enabled
        blacklist = self.get_blacklist(eligible_ends) if self.blacklist_rounds > 0 else set()

        # Build priority lists using availability tracker
        priority_ends, remaining_ends = self.build_priority_lists(
            eligible_ends, cur_time, round_duration_hint, blacklist, trainer_unavail_list or []
        )

        logger.info(
            f"Priority split: {len(priority_ends)} priority, "
            f"{len(remaining_ends)} remaining, {len(blacklist)} blacklisted"
        )

        # Select based on priority mode
        if self.avail_priority == 0:
            # No priority: use standard Oort on all available ends
            all_candidates = set(priority_ends + remaining_ends)
            selected = self._select_with_oort_ucb(
                eligible_ends, all_candidates, num_to_select, round_num
            )

        elif self.avail_priority == 1:
            # Fill mode: prioritize high-priority, fill remaining from others
            selected = self._select_priority_fill(
                eligible_ends, priority_ends, remaining_ends, num_to_select, round_num
            )

        elif self.avail_priority == 2:
            # Strict mode: only select from high-priority clients
            available_priority = min(len(priority_ends), num_to_select)
            logger.info(
                f"Strict priority mode: selecting {available_priority} from "
                f"{len(priority_ends)} priority ends"
            )
            selected = self._select_with_oort_ucb(
                eligible_ends, set(priority_ends), available_priority, round_num
            )

        else:
            logger.warning(
                f"Unknown avail_priority={self.avail_priority}, using mode 0"
            )
            all_candidates = set(priority_ends + remaining_ends)
            selected = self._select_with_oort_ucb(
                eligible_ends, all_candidates, num_to_select, round_num
            )

        newly_selected = set(selected)
        self.newly_selected_this_round = newly_selected
        self.selected_ends = self.selected_ends | newly_selected

        logger.info(
            f"[REFL_SELECT] round {round_num}: selected {len(newly_selected)} new, "
            f"in-flight total {len(self.selected_ends)}"
        )

        self.round = round_num
        self.update_exploration_factor()

        for end_id in selected:
            if end_id in ends:
                count = ends[end_id].get_property(PROP_SELECTED_COUNT) or 0
                ends[end_id].set_property(PROP_SELECTED_COUNT, count + 1)

        # Emit selector-decision telemetry (same schema as oort/feddance) so the
        # mis-selection oracle can score REFL too.
        self.emit_selection(
            round_num, task_to_perform, ends, eligible_ends.keys(), newly_selected,
            per_trainer_extra=getattr(self, "_audit_components", None),
            extra={
                "avail_priority": self.avail_priority,
                "num_priority": len(priority_ends),
                "num_blacklist": len(blacklist),
                "exploration_factor": self.exploration_factor,
            },
        )
        return {key: None for key in newly_selected}

    def build_priority_lists(
        self,
        ends: Dict[str, End],
        cur_time: float,
        round_duration: float,
        blacklist: Set[str],
        unavail_list: List[str],
    ):
        """
        Build priority and remaining client lists based on availability.

        Args:
            ends: All available ends
            cur_time: Current virtual time
            round_duration: Estimated duration of round
            blacklist: Set of blacklisted end IDs
            unavail_list: List of currently unavailable trainers

        Returns:
            Tuple of (priority_end_ids, remaining_end_ids)
        """
        if not self.avail_tracker or self.avail_priority == 0:
            # No availability tracking, all clients have equal priority
            available = [
                eid
                for eid in ends.keys()
                if eid not in blacklist and eid not in unavail_list
            ]
            return [], available

        # Get all available (online) clients
        available_end_ids = [
            eid
            for eid in ends.keys()
            if eid not in blacklist and eid not in unavail_list
        ]

        # Split by priority using availability tracker
        priority_ends, remaining_ends = self.avail_tracker.split_by_priority(
            available_end_ids,
            cur_time,
            round_duration,
            lookup_timeslots=2,
            accuracy=self.avail_probability,
        )

        return priority_ends, remaining_ends

    def _select_priority_fill(
        self,
        ends: Dict[str, End],
        priority_ends: List[str],
        remaining_ends: List[str],
        num_to_select: int,
        round_num: int,
    ) -> List[str]:
        """
        Select clients prioritizing high-priority, fill remaining from others.

        Args:
            ends: All available ends
            priority_ends: High-priority end IDs
            remaining_ends: Remaining end IDs
            num_to_select: Total number to select
            round_num: Current round number

        Returns:
            List of selected end IDs
        """
        selected = []

        # First, select from priority clients
        if priority_ends:
            num_from_priority = min(len(priority_ends), num_to_select)
            priority_selected = self._select_with_oort_ucb(
                ends, set(priority_ends), num_from_priority, round_num
            )
            selected.extend(priority_selected)

        # Fill remaining slots from other clients
        remaining_slots = num_to_select - len(selected)
        if remaining_slots > 0 and remaining_ends:
            remaining_selected = self._select_with_oort_ucb(
                ends, set(remaining_ends), remaining_slots, round_num
            )
            selected.extend(remaining_selected)

        return selected

    def _select_with_oort_ucb(
        self,
        ends: Dict[str, End],
        candidate_end_ids: Set[str],
        num_to_select: int,
        round_num: int,
    ) -> List[str]:
        """
        Run Oort's UCB-based selection on candidate ends.

        Args:
            ends: All available ends
            candidate_end_ids: Set of candidate end IDs to select from
            num_to_select: Number to select
            round_num: Current round number

        Returns:
            List of selected end IDs
        """
        if not candidate_end_ids or num_to_select == 0:
            return []

        # Filter ends to only candidates
        candidate_ends = {eid: ends[eid] for eid in candidate_end_ids if eid in ends}

        if len(candidate_ends) <= num_to_select:
            return list(candidate_ends.keys())

        # Use parent class's calculate_total_utility and selection logic
        # Build utility list for candidates
        utility_list = []
        for end_id, end in candidate_ends.items():
            stat_util = end.get_property("stat_utility")
            if stat_util is not None:
                utility_list.append({PROP_END_ID: end_id, PROP_UTILITY: stat_util})

        if not utility_list:
            # No utility info, select randomly
            return random.sample(list(candidate_end_ids), num_to_select)

        # Calculate total utility with temporal uncertainty and system utility
        utility_list = self.calculate_total_utility(
            utility_list, candidate_ends, round_num
        )

        # Select top-k with exploration/exploitation
        num_exploit = int(num_to_select * (1 - self.exploration_factor))
        num_explore = num_to_select - num_exploit

        # Sort by utility
        utility_list = sorted(utility_list, key=lambda x: x[PROP_UTILITY], reverse=True)

        # Exploitation: top utility clients
        exploit_clients = [item[PROP_END_ID] for item in utility_list[:num_exploit]]

        # Exploration: random from remaining
        remaining_candidates = [
            item[PROP_END_ID] for item in utility_list[num_exploit:]
        ]
        explore_clients = random.sample(
            remaining_candidates, min(num_explore, len(remaining_candidates))
        )

        selected = exploit_clients + explore_clients

        # Pad with random if needed
        while len(selected) < num_to_select and len(candidate_end_ids) > len(selected):
            remaining = candidate_end_ids - set(selected)
            selected.append(random.choice(list(remaining)))

        return selected[:num_to_select]

    def get_blacklist(self, ends: Dict[str, End]) -> Set[str]:
        """
        Get set of blacklisted end IDs based on selection frequency.

        Args:
            ends: All available ends

        Returns:
            Set of blacklisted end IDs
        """
        if self.blacklist_rounds < 0:
            return set()

        blacklist = []

        # Sort by selection count (descending)
        end_counts = []
        for end_id, end in ends.items():
            count = end.get_property(PROP_SELECTED_COUNT)
            if count and count > self.blacklist_rounds:
                end_counts.append((end_id, count))

        # Sort by count descending
        end_counts.sort(key=lambda x: x[1], reverse=True)

        # Take top blacklist_max_len fraction
        max_blacklist = int(self.blacklist_max_len * len(ends))
        blacklist = [end_id for end_id, _ in end_counts[:max_blacklist]]

        if blacklist:
            logger.debug(f"Blacklisted {len(blacklist)} ends: {blacklist}")

        return set(blacklist)

    def pacer(self) -> None:
        """
        Adaptive pacer mechanism to adjust round_threshold.

        Monitors exploitation utility trends and adjusts round_threshold:
        - If utility is flat (< 10% change): increase threshold (faster clients)
        - If utility is volatile (> 500% change): decrease threshold (more clients)
        """
        if self.pacer_step <= 0:
            return  # Pacer disabled

        # Only run pacer at specified intervals
        if self.round < 2 * self.pacer_step or self.round % self.pacer_step != 0:
            return

        # Calculate utility change over last two pacer windows
        if len(self.exploitation_util_history) < 2 * self.pacer_step:
            return

        history_list = list(self.exploitation_util_history)
        util_last_window = sum(history_list[-2 * self.pacer_step : -self.pacer_step])
        util_current_window = sum(history_list[-self.pacer_step :])

        if util_last_window == 0:
            return

        relative_change = abs(util_current_window - util_last_window) / util_last_window

        # Flat utility: increase threshold (prefer faster clients)
        if relative_change <= 0.1:
            old_threshold = self.round_threshold
            self.round_threshold = min(100.0, self.round_threshold + self.pacer_delta)
            logger.info(
                f"Pacer: Utility flat ({relative_change:.2%}), "
                f"increasing threshold {old_threshold}% -> {self.round_threshold}%"
            )

        # Volatile utility: decrease threshold (include more clients)
        elif relative_change >= 5.0:
            old_threshold = self.round_threshold
            self.round_threshold = max(
                self.pacer_delta, self.round_threshold - self.pacer_delta
            )
            logger.info(
                f"Pacer: Utility volatile ({relative_change:.2%}), "
                f"decreasing threshold {old_threshold}% -> {self.round_threshold}%"
            )

        self.last_pacer_round = self.round
