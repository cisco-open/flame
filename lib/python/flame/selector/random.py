# Copyright 2022 Cisco Systems, Inc. and its affiliates
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
"""RandomSelector class."""

import logging
import random
import time
from collections import deque
import numpy as np

from ..common.typing import Scalar
from ..end import End
from . import AbstractSelector, SelectorReturnType
from flame.channel import (
    KEY_CH_SELECT_REQUESTER,
    KEY_CH_STATE,
    VAL_CH_STATE_HTBT_RECV,
    VAL_CH_STATE_HTBT_SEND,
    VAL_CH_STATE_RECV,
    VAL_CH_STATE_SEND,
)
from flame.end import KEY_END_STATE, VAL_END_STATE_NONE, VAL_END_STATE_RECVD, End

logger = logging.getLogger(__name__)


class RandomSelector(AbstractSelector):
    """A random selector class."""

    def __init__(self, **kwargs):
        """Initailize instance."""
        super().__init__(**kwargs)

        try:
            self.k = kwargs["k"]
        except KeyError:
            raise KeyError("k is not specified in config")

        if self.k < 0:
            self.k = 1

        self.round = 0

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
            "util_last_50", "util_last_100", "util_last_200",
            "speed_last_50", "speed_last_100", "speed_last_200",
            "round_last_50", "round_last_100", "round_last_200"
        ]

        for task in tasks:
            for metric in metrics:
                values = self._selector_stats[task]['data'].get(metric, [])
                key = f"stat_{metric}" if "util" in metric else metric
                self._selector_stats[task]["summary"][key] = compute_summary(values)
        
    def _reset_selector_stats(self) -> None:
        self._selector_stats = {}
    
    def select(
        self,
        ends: dict[str, End],
        channel_props: dict[str, Scalar],
        task_to_perform: str = "train",
    ) -> SelectorReturnType:
        """Return k number of ends from the given ends."""
        logger.debug("calling random select")
        # self.requester = channel_props[KEY_CH_SELECT_REQUESTER] if
        # self.requester not in self.selected_ends:
        #     self.selected_ends[self.requester] = set()

        # default, availability unaware way of using ends
        eligible_ends = ends

        logger.debug(f"len(ends), self.k: {len(ends)}, {self.k}")
        # trainers
        if len(ends) < self.k:
            logger.debug(f"not enough ends, need atleast {self.k}")
            time.sleep(8*4)
            return {}

        k = min(len(ends), self.k)
        if k == 0:
            logger.debug("ends is empty")
            return {}
        logger.debug(f"new k = {k}")
        if "round" in channel_props:
            round = channel_props["round"]
        else:
            round = 0
            logger.warning(f"round not found in channel_props: {channel_props}")
        

        if len(self.selected_ends) == 0 or round > self.round:
            logger.info(
                f"let's select {k} ends for new round {round}, task: {task_to_perform}"
            )

            # Return existing selected end_ids if the round did not
            # proceed
            if round <= self.round and len(self.selected_ends) != 0:
                return {key: None for key in self.selected_ends}

            # Adopted from FwdLLM: For each variance comparison, we
            # are selecting the same clients each round TODO (DG/NRL):
            # Seeding needs to be put behind a config param incase
            # another selector doesnt need this? TODO (DG/NRL): How
            # will this work with Unavailability? Will be severly
            # disadvantaged with unavl. Naive solution for this?
            random.seed(round)
            self.selected_ends = set(random.sample(list(ends), k))
            self.round = round

        logger.info(f"selected ends: {self.selected_ends} for round {round} and self.round: {self.round}")

        return {key: None for key in self.selected_ends}

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

        selected_ends = self.selected_ends

        num_ends_to_remove = min(len(self.ordered_updates_recv_ends), self.k)
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
                    logger.debug(
                        f"no end id {end_id} in ends, removing "
                        f"from selected_ends and all_selected"
                    )
                    # NOTE: it is not a guarantee that selected_ends
                    # will still contain the end_id. Thats because it
                    # might have got disconnected/ rejoined in the
                    # middle of a round
                    if end_id in selected_ends:
                        selected_ends.remove(end_id)
                        logger.debug(
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
        pass
        # logger.debug( f"Going to cleanup selector state for "
        #     f"end_id {end_id} since it has left the channel" ) if
        #     (end_id in self.all_selected) and ( end_id not in
        # self.ordered_updates_recv_ends ): # remove end from
        # all_selected if we havent got an update # from it yet. It
        #     would have flushed the agg-weights after # initiating
        # channel.leave(). logger.debug( f"Removing end_id {end_id}
        #     from all_selected" f" since no update received before it
        #     left the channel." ) selected_ends =
        #     self.selected_ends[self.requester] if end_id in
        #     selected_ends: selected_ends.remove(end_id)
        #         logger.debug(f"Also removing end_id {end_id} from
        #         selected_ends") self.selected_ends[self.requester] =
        #     selected_ends

        #     # Track trainers that were sent weights but dropped off
        #     # before sending back an update
        #     if end_id in self.track_selected_trainers_which_left:
        #         self.track_selected_trainers_which_left[end_id] += 1
        #     else: self.track_selected_trainers_which_left[end_id] =
        #         1

        #     total_trainers_dropped_off = 0 for k, v in
        #     self.track_selected_trainers_which_left.items():
        #         total_trainers_dropped_off += v

        #     logger.debug(
        #         f"Trainer: {end_id} with count "
        #         f"{self.track_selected_trainers_which_left[end_id]}, left "
        #         f"before returning update. "
        #         f"total_trainers_dropped_off: {total_trainers_dropped_off} "
        #         f"self.track_selected_trainers_which_left: "
        #         f"{self.track_selected_trainers_which_left}"
        #     )
        #     if end_id in self.all_selected.keys():
        #         del self.all_selected[end_id]
        # elif (end_id in self.all_selected) and ( end_id in
        #     self.ordered_updates_recv_ends ): # Dont remove it if it
        # was in all_selected and we have got # an update from it
        #     before it did channel.leave(). It has # completed its
        #     participation for this round. logger.debug( f"Update was
        #     alreacy received from {end_id} before it left " f"the
        #     channel. Not deleting from all_ends now." ) else:
        #         logger.warn( f"End_id {end_id} remove check from
        #         all_selected failed. " f"Need to check" )

    def remove_from_selected_ends(self, ends: dict[str, End], end_id: str) -> None:
        """Remove an end from selected ends"""
        pass
        # selected_ends = self.selected_ends[self.requester] if end_id
        # in ends.keys(): if end_id in selected_ends: logger.debug(
        #     f"Going to remove end_id {end_id} from selected_ends "
        #         f"{selected_ends}" ) selected_ends.remove(end_id)
        #             self.selected_ends[self.requester] =
        #             selected_ends logger.debug(
        #         f"self.selected_ends: {self.selected_ends} after "
        #         f"removing end_id: {end_id}" ) else: logger.debug(
        #         f"Attempted to remove end {end_id} from "
        #         f"self.selected_ends {self.selected_ends}, but it
        #             wasnt present" ) else: logger.debug( f"Attempted
        #             to remove end {end_id} from "
        #         f"self.selected_ends {self.selected_ends}, but it
        #     wasnt in ends")
