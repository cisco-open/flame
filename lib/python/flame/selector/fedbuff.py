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
"""FedBuffSelector class."""

import logging
import random
import time
from collections import deque
import numpy as np

from flame.channel import (
    KEY_CH_SELECT_REQUESTER,
    KEY_CH_STATE,
    VAL_CH_STATE_HTBT_RECV,
    VAL_CH_STATE_HTBT_SEND,
    VAL_CH_STATE_RECV,
    VAL_CH_STATE_SEND,
)
from flame.common.typing import Scalar
from flame.end import KEY_END_STATE, VAL_END_STATE_NONE, VAL_END_STATE_RECVD, End
from flame.selector import AbstractSelector, SelectorReturnType

from flame.selector.properties import (
    PROP_LAST_EVAL_ROUND,
    PROP_CLIENT_TASK_TRAIN_DURATION,
    PROP_STAT_UTILITY,
)

logger = logging.getLogger(__name__)

SEND_TIMEOUT_WAIT_S = 90


class FedBuffSelector(AbstractSelector):
    """A selector class for fedbuff-based asyncfl."""

    def __init__(self, **kwargs):
        """Initailize instance."""
        super().__init__(**kwargs)

        try:
            self.c = kwargs["c"]
        except KeyError:
            raise KeyError("c (concurrency level) is not specified in config")

        try:
            self.agg_goal = kwargs["aggGoal"]
        except KeyError:
            raise KeyError("agg_goal (aggregation goal) is not specified in config")

        # Tracking selected ends to ensure selection correctness for
        # each round (a trainer can participate only once per round).
        self.all_selected = dict()
        self.selected_ends = dict()

        # Tracks updates received from trainers and makes them
        # available to select again
        self.ordered_updates_recv_ends = list()

        # Tracks timeouted trainers and number of times it happened to
        # a trainer
        self.track_trainer_timeouts = dict()

        # Tracks trainers that were selected but left training in
        # between
        self.track_selected_trainers_which_left = dict()

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
        trainer_unavail_list: list = None,
        **kwargs,
    ) -> SelectorReturnType:
        """Select ends from the given ends to meet concurrency level.

        This select method chooses ends differently depending on what
        state a channel is in. In 'send' state, it chooses ends that
        are not in self.selected_ends. In 'recv' state, it chooses all
        ends from self.selected_ends. Essentially, if an end is in
        self.selected_ends, it means that we sent some message already
        to that end. For such an end, we exclude it from send and
        include it for recv in return.
        """
        # TODO (DG): The recv state should also only select from ends
        # IF an update doesnt already exist in the cache
        logger.debug("calling fedbuff select")
        concurrency = min(len(ends), self.c)
        logger.info(f"len(ends): {len(ends)}, c: {self.c}, concurrency: {concurrency}")

        if concurrency == 0:
            logger.debug("ends is empty")
            return {}

        if KEY_CH_STATE not in channel_props:
            raise KeyError(f"channel property doesn't have {KEY_CH_STATE}")

        self.requester = channel_props[KEY_CH_SELECT_REQUESTER]
        if self.requester not in self.selected_ends:
            self.selected_ends[self.requester] = set()

        # default, availability unaware way of using ends
        eligible_ends = ends

        # Make a filter of unavailable ends, update eligible_ends
        # given trainer_unavail_list
        if trainer_unavail_list != [] and trainer_unavail_list is not None:
            # Updating passed ends and filtering out unavailable ones
            # before passing
            eligible_ends = {
                end_id: end
                for end_id, end in ends.items()
                if end_id not in trainer_unavail_list
            }
            logger.debug(
                f"Fedbuff select got non-empty trainer_unavail_list, "
                f"populated eligible_ends: {eligible_ends}"
            )

        results = {}
        if channel_props[KEY_CH_STATE] == VAL_CH_STATE_SEND:
            results = self._handle_send_state(
                eligible_ends, concurrency, connected_ends=ends
            )  # Challenge 13: full pool for cleanup
            self.emit_selection(
                channel_props.get("round", 0),
                "train",
                ends,
                eligible_ends.keys(),
                list(results.keys()),
                extra={"concurrency": concurrency, "requester": self.requester,
                       "vclock_now": channel_props.get("vclock_now")},
            )

        elif channel_props[KEY_CH_STATE] == VAL_CH_STATE_RECV:
            # TODO: (DG) See if eligible_ends should be passed here
            # too in place of ends
            results = self._handle_recv_state(ends, concurrency)

        elif channel_props[KEY_CH_STATE] == VAL_CH_STATE_HTBT_RECV:
            results = self._handle_htbt_recv_state(ends)

        elif channel_props[KEY_CH_STATE] == VAL_CH_STATE_HTBT_SEND:
            results = self._handle_htbt_send_state(ends)

        else:
            state = channel_props[KEY_CH_STATE]
            raise ValueError(f"unkown channel state: {state}")

        logger.debug(
            f"requester: {self.requester}, selected ends: {self.selected_ends}"
        )
        logger.debug(
            f"channel state: {channel_props[KEY_CH_STATE]}, results: {results}"
        )

        return results

    def _cleanup_recvd_ends(self, ends: dict[str, End]):
        """Clean up ends whose a message was received, from selected
        ends.

        NOTE: It sets the end state to none which makes it eligible to
        be sampled again. This can cause problems if sampled in the
        same round. Thus, for aggregator, the _cleanup_recvd_ends
        should be triggered only after aggregation of weights succeeds
        on meeting agg_goal."""
        logger.debug("clean up recvd ends")
        logger.debug(f"ends: {ends.keys()}")
        logger.debug(f"selected ends: {self.selected_ends}")

        selected_ends = self.selected_ends[self.requester]
        logger.debug(
            f"self.requester: {self.requester} and selected_ends: "
            f"{selected_ends} before processing"
        )

        num_ends_to_remove = min(len(self.ordered_updates_recv_ends), self.agg_goal)
        logger.debug(f"num_ends_to_remove: {num_ends_to_remove}")
        if num_ends_to_remove != 0:
            ends_to_remove = self.ordered_updates_recv_ends[:num_ends_to_remove]
            logger.debug(
                f"Will remove these ends from "
                f"ordered_updates_recv_ends: {ends_to_remove}"
                f" and selected_ends and all_selected"
            )

            # removing the first agg-goal number of ends to free them
            # to participate in the next round
            self.ordered_updates_recv_ends = self.ordered_updates_recv_ends[
                num_ends_to_remove:
            ]
            logger.debug(
                f"self.ordered_updates_recv_ends after removing first "
                f"num_ends_to_remove: {num_ends_to_remove} "
                f"elements: {self.ordered_updates_recv_ends}"
            )

            for end_id in ends_to_remove:
                if end_id not in ends:
                    # something happened to end of end_id (e.g.,
                    # connection loss) let's remove it from
                    # selected_ends
                    logger.info(
                        f"no end id {end_id} in ends, removing "
                        f"from selected_ends and all_selected"
                    )
                    # NOTE: it is not a guarantee that selected_ends
                    # will still contain the end_id. Thats because it
                    # might have got disconnected/ rejoined in the
                    # middle of a round
                    if end_id in selected_ends:
                        selected_ends.remove(end_id)
                        logger.info(
                            f"No end id {end_id} in ends, removed from "
                            f"selected_ends: "
                            f"{selected_ends}"
                        )
                    if end_id in self.all_selected:
                        del self.all_selected[end_id]
                        logger.debug(
                            f"No end id {end_id} in ends, removed from "
                            f"self.all_selected: {self.all_selected}"
                        )
                else:
                    state = ends[end_id].get_property(KEY_END_STATE)
                    logger.debug(
                        f"End_id {end_id} found in selected_ends in state: {state}, "
                        f"selected_ends: {selected_ends} and self.all_selected: "
                        f"{self.all_selected}"
                    )
                    if state == VAL_END_STATE_RECVD:
                        ends[end_id].set_property(KEY_END_STATE, VAL_END_STATE_NONE)
                        logger.debug(
                            f"Setting {end_id} state to {VAL_END_STATE_NONE}, "
                            f"and removing from selected_ends "
                            f"and all_selected"
                        )
                        if end_id in selected_ends:
                            selected_ends.remove(end_id)
                            logger.debug(
                                f"FOUND end id {end_id} in state: {state}.. "
                                f"removed from "
                                f"selected_ends: {selected_ends}"
                            )
                        if end_id in self.all_selected:
                            del self.all_selected[end_id]
                            logger.debug(
                                f"FOUND end id {end_id} in state: {state}.. "
                                f"removed from "
                                f"self.all_selected: "
                                f"{self.all_selected}"
                            )
                    elif state == VAL_END_STATE_NONE:
                        # TODO: (DG) Recheck if it needs to be deleted
                        # from here as well. Is the failure scenario
                        # being handled correctly if the trainer
                        # contributes, fails and then comes back
                        # within the same round. TODO: (DG) Need a
                        # diagram in the paper to explain this?
                        logger.debug(
                            f"Found end {end_id} in state {VAL_END_STATE_NONE}. Might have "
                            f"left/rejoined. Need to remove it from "
                            f"selected_ends and self.all_selected if it "
                            f"was selected"
                        )
                        if end_id in selected_ends:
                            selected_ends.remove(end_id)
                            logger.debug(
                                f"FOUND end id {end_id} in state: {state}.. "
                                f"removed from "
                                f"selected_ends: {selected_ends}"
                            )
                        if end_id in self.all_selected:
                            del self.all_selected[end_id]
                            logger.debug(
                                f"FOUND end id {end_id} in state: {state}.. "
                                f"removed from "
                                f"self.all_selected: "
                                f"{self.all_selected} too"
                            )
                    else:
                        logger.debug(
                            f"FOUND end id {end_id} in state: {state}. "
                            f"Not doing anything"
                        )
        else:
            logger.debug("No ends to remove so far")

    def _cleanup_removed_ends(self, end_id):
        logger.debug(
            f"Going to cleanup selector state for "
            f"end_id {end_id} since it has left the channel"
        )
        if (end_id in self.all_selected) and (
            end_id not in self.ordered_updates_recv_ends
        ):
            # remove end from all_selected if we havent got an update
            # from it yet. It would have flushed the agg-weights after
            # initiating channel.leave().
            logger.debug(
                f"Removing end_id {end_id} from all_selected"
                f" since no update received before it left the channel."
            )
            selected_ends = self.selected_ends[self.requester]
            if end_id in selected_ends:
                selected_ends.remove(end_id)
                logger.debug(f"Also removing end_id {end_id} from selected_ends")
                self.selected_ends[self.requester] = selected_ends

            # Track trainers that were sent weights but dropped off
            # before sending back an update
            if end_id in self.track_selected_trainers_which_left:
                self.track_selected_trainers_which_left[end_id] += 1
            else:
                self.track_selected_trainers_which_left[end_id] = 1

            total_trainers_dropped_off = 0
            for k, v in self.track_selected_trainers_which_left.items():
                total_trainers_dropped_off += v

            logger.debug(
                f"Trainer: {end_id} with count "
                f"{self.track_selected_trainers_which_left[end_id]}, left "
                f"before returning update. "
                f"total_trainers_dropped_off: {total_trainers_dropped_off} "
                f"self.track_selected_trainers_which_left: "
                f"{self.track_selected_trainers_which_left}"
            )
            if end_id in self.all_selected.keys():
                del self.all_selected[end_id]
        elif (end_id in self.all_selected) and (
            end_id in self.ordered_updates_recv_ends
        ):
            # Dont remove it if it was in all_selected and we have got
            # an update from it before it did channel.leave(). It has
            # completed its participation for this round.
            logger.debug(
                f"Update was alreacy received from {end_id} before it left "
                f"the channel. Not deleting from all_ends now."
            )
        else:
            logger.warn(
                f"End_id {end_id} remove check from all_selected failed. "
                f"Need to check"
            )

    def _cleanup_send_ends(self):
        # TODO: (DG) Get a more principled solution here. Hacky right
        # now to fix the issue for trainer side when failures occur.

        # NOTE: (DG) This function isn't incorporated into async_oort
        # since the trainer still uses fedbuff selector and not
        # async_oort selector

        logger.debug("Going to cleanup selector state after " "send.")
        selected_ends = self.selected_ends[self.requester]
        curr_list_selected_ends = list(selected_ends)
        for end_id in curr_list_selected_ends:
            logger.debug(f"Removing end_id {end_id} from selected_ends")
            selected_ends.remove(end_id)
            self.selected_ends[self.requester] = selected_ends
            if end_id in self.all_selected:
                logger.debug(
                    f"Also removing end_id {end_id} from " f"self.all_selected"
                )
                del self.all_selected[end_id]

    def _handle_send_state(
        self, ends: dict[str, End], concurrency: int,
        connected_ends: dict[str, End] = None,
    ) -> SelectorReturnType:
        selected_ends = self.selected_ends[self.requester]

        # Challenge 13: cleanup must check CONNECTED membership, not availability-
        # eligibility — an in-flight trainer that merely went UN_AVL (or is the
        # wrong task-type) is absent from the filtered `ends` but still connected;
        # removing it makes the aggregator forget it is waiting. An empty eligible
        # pool would otherwise wipe ALL shared selected_ends → hang. Use the full
        # connected pool when provided; fall back to `ends` for backward compat.
        _connected = connected_ends if connected_ends is not None else ends
        # Check for invalid selections and remove them
        for end_id in list(selected_ends):
            if end_id not in _connected:
                # something happened to end of end_id (e.g.,
                # connection loss) let's remove it from selected_ends
                # so that you can fill that spot with another trainer
                logger.info(
                    f"Removing invalid prior selection! "
                    f"No end id {end_id} in ends, "
                    f"removing from selected_ends. "
                    f"NOT from all_selected right now "
                    f"cause aggregation for that "
                    f"round hasnt completed yet"
                )
                selected_ends.remove(end_id)
                # NOTE: Not removing end_id from all_selected since it
                # might have already participated in the same round
                # (if it is still in all_ends)

        extra = max(0, concurrency - len(selected_ends))

        logger.debug(
            f"c: {concurrency}, ends: {ends.keys()},"
            f"len(selected_ends): {len(selected_ends)}, extra: {extra}"
        )
        candidates = []
        idx = 0

        # Randomized sampling over the candidate ends. Do NOT reseed the global
        # RNG here: a per-call `random.seed(time.time())` made selection
        # non-reproducible, clobbered the process-global random state for every
        # other consumer, and defeated the aggregator's deterministic seed
        # (breaking real/sim parity). The RNG is seeded once at aggregator init.
        shuffled_end_ids = list(ends.keys())  # get the keys
        logger.debug(f"Original shuffled_end_ids: {shuffled_end_ids}")
        self._pyrng.shuffle(shuffled_end_ids)  # then shuffle (dedicated RNG)
        logger.debug(f"Updated shuffled_end_ids: {shuffled_end_ids}")

        # Invalidate previous all_selected entry if you don't get an
        # update in UPDATE_TIMEOUT_WAIT_S. The client might have
        # dropped the message with transient unavailability.

        # TODO: (DG) Check if it is still needed after
        # cleanup_remove_end() method
        curr_all_selected_ends = list(self.all_selected.keys())
        for end in curr_all_selected_ends:
            current_time_s = time.time()
            if end in self.all_selected.keys():
                # Check again to avoid possible case of race condition
                # when all_selected has been updated from another
                # thread
                trainer_weight_send_timestamp_s = self.all_selected[end]
                if (
                    trainer_weight_send_timestamp_s
                    < (current_time_s - SEND_TIMEOUT_WAIT_S)
                ) and (end not in self.ordered_updates_recv_ends):
                    # trainer hasn't returned with an update in
                    # SEND_TIMEOUT_WAIT_S delete it from
                    # self.all_selected so that it is eligible to be
                    # sampled again
                    logger.info(
                        f"Removing end {end} from self.all_selected "
                        f"since havent "
                        f"got its update in {SEND_TIMEOUT_WAIT_S}"
                    )

                    # Tracking timeouts and time spend waiting TODO:
                    # (DG) Check if it is okay to have it triggered
                    # for oracular too? TODO: (DG) pass
                    # timeout_duration as a flag from config, also
                    # pass enable/disable it?
                    if end in self.track_trainer_timeouts:
                        self.track_trainer_timeouts[end] += 1
                    else:
                        self.track_trainer_timeouts[end] = 1

                    # Capture total time spent in timeouts
                    num_of_timeouts_occured = 0
                    for k, v in self.track_trainer_timeouts.items():
                        num_of_timeouts_occured += v

                    total_time_spent_timeouts_s = (
                        num_of_timeouts_occured * SEND_TIMEOUT_WAIT_S
                    )

                    logger.info(
                        f"Timeout for trainer: {end} with count "
                        f"{self.track_trainer_timeouts[end]}. "
                        f"num_of_timeouts_occured : "
                        f"{num_of_timeouts_occured}, "
                        f"total_time_spent_timeouts_s: "
                        f"{total_time_spent_timeouts_s}, "
                        f"Timeout frequency: {self.track_trainer_timeouts}"
                    )

                    # delete the end from self.all_selected AND from
                    # selected_ends -- extra (computed above) is derived
                    # from len(selected_ends), so a reclaim that only
                    # touches all_selected leaves the concurrency slot
                    # stuck occupied (see async_oort.py's identical fix
                    # and examples/MIGRATING_TO_LAUNCHER.md's aggregator
                    # gotchas for the fwdllm deadlock this class of bug
                    # caused).
                    if end in self.all_selected.keys():
                        del self.all_selected[end]
                    selected_ends.discard(end)

        for end_id in shuffled_end_ids:
            if end_id in self.all_selected.keys():
                # skip if an end is already selected
                logger.debug(f"end_id {end_id} in all_selected, so skipping")
                continue

            idx += 1
            if len(candidates) < extra:
                candidates.append(end_id)
                logger.debug(f"Added end_id: {end_id} to candidates: {candidates}")
                continue

            i = self._pyrng.randrange(idx)
            if i < extra:
                candidates[i] = end_id

        logger.debug(f"candidates: {candidates}")

        # add candidates to selected ends
        selected_ends = selected_ends.union(candidates)
        self.selected_ends[self.requester] = selected_ends
        logger.debug(
            f"added candidates to selected_ends: {candidates}, selected_ends: "
            f"{selected_ends}, "
            f"self.selected_ends[req]: {self.selected_ends[self.requester]}"
        )

        for candidate_end in candidates:
            # Add to all_selected. {key: end, val: TS epoch (s)}
            self.all_selected[candidate_end] = time.time()
        logging.debug(
            f"self.all_selected {self.all_selected} after combining with "
            f"candidates {candidates}"
        )

        logger.debug(f"handle_send_state returning candidates: {candidates}")

        # Computations for selector statistics
        self._select_run_counter += 1

        for selected_end_id in candidates:
            task_to_perform = "train"
            end_stat_util = ends[selected_end_id].get_property(PROP_STAT_UTILITY)
            end_speed = ends[selected_end_id].get_property(PROP_CLIENT_TASK_TRAIN_DURATION)
            end_last_round = ends[selected_end_id].get_property(PROP_LAST_EVAL_ROUND)
            # Insert to queues tracking stat_util, speed, round data
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

        return {end_id: None for end_id in candidates}

    def _handle_htbt_send_state(self, ends: dict[str, End]) -> SelectorReturnType:
        # TODO: Implement again Earlier (not fully functional)
        # implementation was using same code as handle_send_state()
        # and was commented out.

        logger.debug(f"ends: {ends}")
        return {end_id: None for end_id in ends}

    def _handle_recv_state(
        self, ends: dict[str, End], concurrency: int
    ) -> SelectorReturnType:
        selected_ends = self.selected_ends[self.requester]
        logger.debug(f"selected_ends: {selected_ends}")

        # from the selected ends, remove those that are in recv state
        # already This is done to avoid waiting on trainers that you
        # have already heard from. If selected ends is empty, get()
        # will proceed and wait on distribute_weights before running
        # again. Thus, it avoids stalling and ensures progress
        for end_id in list(selected_ends):
            # trainer might have become unavailable, check if it is
            # still available first
            if end_id in ends:
                if ends[end_id].get_property(KEY_END_STATE) == VAL_END_STATE_RECVD:
                    selected_ends.remove(end_id)
                    logger.debug(
                        f"Removed end_id {end_id} from selected ends since it "
                        f"was already in recvd state"
                    )
                # TODO: (DG) Remove ends from send state also here for
                # the trainer side? But how will it impact the
                # aggregator?
            else:
                # TODO: (DG) Should we not remove it from selected
                # ends here?
                logger.debug(
                    f"Tried to check state of end {end_id} but it is no "
                    f"longer in self._ends"
                )

        if len(selected_ends) == 0:
            logger.debug(f"len(selected_ends)=0, let's select {concurrency} ends")

            candidates = dict()
            for end_id, end in ends.items():
                if end_id not in self.all_selected.keys():
                    logging.debug(
                        f"end_id {end_id} not in all_selected, adding "
                        f"to candidates: key {end_id}, val: {end}"
                    )
                    candidates[end_id] = end

            cc = min(len(candidates), concurrency)
            logger.debug(
                f"Will pick cc: {cc} as min(candidates,concurrency) "
                f"from candidates: {candidates}"
            )
            selected_ends = set(self._pyrng.sample(sorted(candidates), cc))

            self.selected_ends[self.requester] = selected_ends
            logger.debug(
                f"self.selected_ends[req]: {self.selected_ends[self.requester]}"
            )

            for selected_end in selected_ends:
                # Add to all_selected. {key: end, val: TS epoch (s)}
                self.all_selected[selected_end] = time.time()
            logging.debug(
                f"self.all_selected {self.all_selected} after combining with "
                f"selected_ends {selected_ends}"
            )

        logger.debug(f"handle_recv_state returning selected_ends: {selected_ends}")

        return {key: None for key in selected_ends}

    def _handle_htbt_recv_state(self, ends: dict[str, End]) -> SelectorReturnType:
        # TODO: Implement again Earlier (not fully functional)
        # implementation was using same code as handle_recv_state()
        # and was commented out.

        logger.debug(f"ends: {ends}")
        return {key: None for key in ends}

    def reset_end_state_to_none(self, ends: dict[str, End], end_id: str) -> None:
        """Reset's the state of end_id from send/recv to none"""
        if end_id in ends.keys():
            curr_end_state = ends[end_id].get_property(KEY_END_STATE)
            ends[end_id].set_property(KEY_END_STATE, VAL_END_STATE_NONE)
            new_end_state = ends[end_id].get_property(KEY_END_STATE)
            logger.debug(
                f"Successfully reset state for end "
                f"{end_id} from previous: {curr_end_state} to "
                f"current: {new_end_state}"
            )
        else:
            logger.debug(
                f"Attempted to reset end {end_id} state " f"but it wasnt in ends"
            )

    def remove_from_selected_ends(self, ends: dict[str, End], end_id: str) -> None:
        """Remove an end from selected ends"""
        selected_ends = self.selected_ends[self.requester]
        if end_id in ends.keys():
            if end_id in selected_ends:
                logger.debug(
                    f"Going to remove end_id {end_id} from selected_ends "
                    f"{selected_ends}"
                )
                selected_ends.remove(end_id)
                self.selected_ends[self.requester] = selected_ends
                logger.debug(
                    f"self.selected_ends: {self.selected_ends} after "
                    f"removing end_id: {end_id}"
                )
            else:
                logger.debug(
                    f"Attempted to remove end {end_id} from "
                    f"self.selected_ends {self.selected_ends}, but it wasnt present"
                )
        else:
            logger.debug(
                f"Attempted to remove end {end_id} from "
                f"self.selected_ends {self.selected_ends}, but it wasnt in ends"
            )

    def reset_selected_ends(self, requester: str) -> None:
        """Reset mapping between requester and selected ends.

        This is needed when requester leaves channel due to e.g.,
        failure.
        """
        logger.debug(f"trying to reset selected ends of {requester}")
        if requester not in self.selected_ends:
            return

        selected_ends = self.selected_ends[requester]

        # Updated all_selected here when updating in other parts TODO:
        # (DG) Needs to be tested if it works
        all_selected_keys_set = set(self.all_selected.keys())
        diff_of_ends = all_selected_keys_set.difference(selected_ends)
        for end_id in diff_of_ends:
            # Add to all_selected. {key: end, val: TS epoch (s)}
            self.all_selected[end_id] = time.time()

        del self.selected_ends[requester]

        logger.debug(f"all selected: {self.all_selected}")
        logger.debug(f"selected_ends: {self.selected_ends}")
        logger.debug(f"done with resetting selected ends of {requester}")
