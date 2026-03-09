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
from flame.selector.oort import (
    OortSelector,
    PROP_UTILITY,
    PROP_END_ID,
    PROP_SELECTED_COUNT,
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

        # Track utility history for pacer
        self.exploitation_util_history = deque(maxlen=200)
        self.last_pacer_round = 0

        # CRITICAL: Initialize selected_ends as a set to track in-flight trainers
        # For SyncFL with overcommitment, this prevents re-selecting trainers
        # that haven't returned their updates yet
        if not hasattr(self, 'selected_ends'):
            self.selected_ends = set()
        
        # Track which trainers were newly selected in the current round
        # This is needed because select() gets called twice per round:
        # once for distribute, once for aggregate
        if not hasattr(self, 'newly_selected_this_round'):
            self.newly_selected_this_round = set()

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

        # NOTE: Cleanup of ordered_updates_recv_ends and selected_ends is now done
        # in _cleanup_recvd_ends() (inherited from OortSelector) immediately after
        # aggregation completes. This fixes race condition where trainers returning
        # updates between agg_goal and next select() would be incorrectly kept in
        # the in-flight set.

        # Return existing selected end_ids if round did not proceed
        # CRITICAL FIX: Return only newly selected trainers from this round,
        # not the cumulative in-flight set. select() is called twice per round:
        # once for distribute (which we want to remember), once for aggregate (which should return same set)
        if round_num <= self.round and hasattr(self, 'newly_selected_this_round') and len(self.newly_selected_this_round) != 0:
            logger.info(f"[RETURN_CACHED] Round {round_num}: Returning {len(self.newly_selected_this_round)} cached newly selected trainers")
            return {key: None for key in self.newly_selected_this_round}

        # Run pacer to adjust round_threshold
        self.pacer()

        # Filter out unavailable trainers AND trainers that are currently "in flight"
        # (selected in previous rounds but haven't returned updates yet)
        # This is CRITICAL for SyncFL with overcommitment (selecting 1.3K but waiting for K)
        
        # Get trainers that are still in flight from previous rounds  
        in_flight_trainers = self.selected_ends if hasattr(self, 'selected_ends') and isinstance(self.selected_ends, set) else set()
        
        # Get unavailable trainer set for detailed logging
        unavail_set = set(trainer_unavail_list) if trainer_unavail_list else set()
        
        # DETAILED LOGGING: Track filtering at each step
        logger.info(
            f"[FILTER_DEBUG] Round {round_num}: "
            f"Total ends available: {len(ends)}"
        )
        logger.info(
            f"[FILTER_DEBUG] Unavailable ends: {len(unavail_set)} trainers"
        )
        if len(unavail_set) > 0 and len(unavail_set) <= 10:
            logger.info(f"[FILTER_DEBUG] Unavailable IDs (sample): {list(unavail_set)[:10]}")
        
        # DEBUG: Check for specific test trainers
        test_trainer_389 = '505f9fc483cf4df68a2409257b5fad7d3c580389'
        test_trainer_411 = '505f9fc483cf4df68a2409257b5fad7d3c580411'
        
        trainer_389_in_ends = test_trainer_389 in ends
        trainer_389_in_inflight = test_trainer_389 in in_flight_trainers
        trainer_389_in_unavail = test_trainer_389 in unavail_set
        
        trainer_411_in_ends = test_trainer_411 in ends
        trainer_411_in_inflight = test_trainer_411 in in_flight_trainers
        trainer_411_in_unavail = test_trainer_411 in unavail_set
        
        logger.info(
            f"[TRACK_SELECT] Round {round_num}: "
            f"389: in_ends={trainer_389_in_ends}, in_flight={trainer_389_in_inflight}, unavail={trainer_389_in_unavail} | "
            f"411: in_ends={trainer_411_in_ends}, in_flight={trainer_411_in_inflight}, unavail={trainer_411_in_unavail} | "
            f"selected_ends_size={len(self.selected_ends) if hasattr(self, 'selected_ends') else 0}"
        )
        
        logger.info(
            f"[FILTER_DEBUG] In-flight trainers: {len(in_flight_trainers)} trainers"
        )
        if len(in_flight_trainers) > 0 and len(in_flight_trainers) <= 10:
            logger.info(f"[FILTER_DEBUG] In-flight IDs (sample): {list(in_flight_trainers)[:10]}")
        elif len(in_flight_trainers) > 10:
            logger.info(f"[FILTER_DEBUG] In-flight IDs (first 10): {list(in_flight_trainers)[:10]}")
        
        # Log ordered_updates_recv_ends for debugging cleanup
        if hasattr(self, 'ordered_updates_recv_ends'):
            logger.info(
                f"[FILTER_DEBUG] Received updates (pending cleanup): {len(self.ordered_updates_recv_ends)} trainers"
            )
            if len(self.ordered_updates_recv_ends) > 0:
                logger.info(
                    f"[FILTER_DEBUG] Received update IDs (pending): {list(self.ordered_updates_recv_ends)[:10]}"
                )
        
        eligible_ends = {
            end_id: end
            for end_id, end in ends.items()
            if (not trainer_unavail_list or end_id not in trainer_unavail_list)
            and end_id not in in_flight_trainers  # NEW: filter out in-flight trainers
        }

        # DEBUG: Check if test trainers made it through filtering
        trainer_389_in_eligible = test_trainer_389 in eligible_ends
        trainer_411_in_eligible = test_trainer_411 in eligible_ends
        
        logger.info(
            f"[TRACK_SELECT] Round {round_num}: After filtering - "
            f"389: eligible={trainer_389_in_eligible} | 411: eligible={trainer_411_in_eligible}"
        )
        
        # Check for bugs
        if trainer_389_in_inflight and trainer_389_in_eligible:
            logger.error(
                f"[BUG_FOUND] Round {round_num}: Trainer 389 is in BOTH in_flight and eligible_ends! "
                f"This should never happen!"
            )
        if trainer_411_in_inflight and trainer_411_in_eligible:
            logger.error(
                f"[BUG_FOUND] Round {round_num}: Trainer 411 is in BOTH in_flight and eligible_ends! "
                f"This should never happen!"
            )

        logger.info(
            f"[FILTER_DEBUG] Eligible ends after filtering: {len(eligible_ends)} out of {len(ends)} "
            f"(filtered out: unavail={len(unavail_set)}, in_flight={len(in_flight_trainers)})"
        )
        
        # Log a few sample eligible end IDs for verification
        if len(eligible_ends) > 0:
            sample_eligible = list(eligible_ends.keys())[:5]
            logger.info(f"[FILTER_DEBUG] Sample eligible end IDs: {sample_eligible}")

        if len(eligible_ends) == 0:
            logger.error(
                f"[SELECTION_FAILED] Round {round_num}: NO eligible trainers! "
                f"total_ends={len(ends)}, unavail={len(unavail_set)}, in_flight={len(in_flight_trainers)}"
            )
            return {}

        # Adjust selection count based on available eligible ends
        num_to_select = min(num_of_ends, len(eligible_ends))
        
        # CRITICAL: Warn if we cannot select enough trainers
        if num_to_select < num_of_ends:
            shortage = num_of_ends - num_to_select
            logger.warning(
                f"[SELECTION_SHORTAGE] Round {round_num}: Can only select {num_to_select}/{num_of_ends} trainers. "
                f"Shortage of {shortage} trainers. "
                f"Breakdown: total={len(ends)}, unavail={len(unavail_set)}, in_flight={len(in_flight_trainers)}, "
                f"eligible={len(eligible_ends)}"
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

        # Store selected ends as a set
        old_in_flight = self.selected_ends if hasattr(self, 'selected_ends') else set()
        newly_selected = set(selected)
        
        # CRITICAL: Store the newly selected set for this round
        # This is returned on subsequent select() calls in the same round (for aggregate)
        self.newly_selected_this_round = newly_selected
        
        # Add newly selected trainers to existing in-flight ones instead of replacing
        self.selected_ends = old_in_flight | newly_selected
        
        # Log selection summary for tracking
        logger.info(
            f"[SELECTION_SUMMARY] Round {round_num}:"
        )
        logger.info(
            f"  - Newly selected this round: {len(newly_selected)} trainers"
        )
        if len(newly_selected) <= 10:
            logger.info(f"    IDs: {list(newly_selected)}")
        else:
            logger.info(f"    IDs (first 10): {list(newly_selected)[:10]}")
        
        if len(old_in_flight) > 0:
            logger.info(
                f"  - Still in-flight from previous rounds: {len(old_in_flight)} trainers"
            )
            if len(old_in_flight) <= 10:
                logger.info(f"    IDs: {list(old_in_flight)}")
            else:
                logger.info(f"    IDs (first 10): {list(old_in_flight)[:10]}")
        
        logger.info(
            f"  - Total tracked in-flight: {len(self.selected_ends)} trainers"
        )
        
        # DEBUG: Check if test trainers were selected
        test_trainer_389 = '505f9fc483cf4df68a2409257b5fad7d3c580389'
        test_trainer_411 = '505f9fc483cf4df68a2409257b5fad7d3c580411'
        
        trainer_389_newly_selected = test_trainer_389 in newly_selected
        trainer_411_newly_selected = test_trainer_411 in newly_selected
        
        if trainer_389_newly_selected or trainer_411_newly_selected:
            logger.info(
                f"[TRACK_SELECT] Round {round_num}: Selection result - "
                f"389: selected={trainer_389_newly_selected} | 411: selected={trainer_411_newly_selected}"
            )
        
        # Check for bugs
        if test_trainer_389 in newly_selected and test_trainer_389 in old_in_flight:
            logger.error(
                f"[BUG_FOUND] Round {round_num}: Trainer 389 was SELECTED despite being in old_in_flight! "
                f"This should have been filtered out!"
            )
        if test_trainer_411 in newly_selected and test_trainer_411 in old_in_flight:
            logger.error(
                f"[BUG_FOUND] Round {round_num}: Trainer 411 was SELECTED despite being in old_in_flight! "
                f"This should have been filtered out!"
            )
        
        # Update round tracking
        self.round = round_num

        # Update exploration factor
        self.update_exploration_factor()

        # Increment selection count for selected ends
        for end_id in selected:
            if end_id in ends:
                count = ends[end_id].get_property(PROP_SELECTED_COUNT)
                if count is None:
                    count = 0
                ends[end_id].set_property(PROP_SELECTED_COUNT, count + 1)

        # CRITICAL FIX: Return only NEWLY selected trainers, not all in-flight trainers
        # self.selected_ends tracks all in-flight trainers (old + new) for filtering in next round
        # But we must only return the newly selected ones to avoid re-sending weights to in-flight trainers
        logger.info(f"[RETURN] Returning {len(newly_selected)} newly selected trainers to caller")
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
