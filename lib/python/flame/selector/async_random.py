"""OortSelector class."""

import logging
import math
import random
import time
from datetime import timedelta
from collections import deque

from flame.config import TrainerAvailState
import numpy as np
from flame.channel import (
    KEY_CH_SELECT_REQUESTER,
    KEY_CH_STATE,
    VAL_CH_STATE_RECV,
    VAL_CH_STATE_SEND,
)
from flame.common.typing import Scalar
from flame.common.util import MLFramework, get_ml_framework_in_use
from flame.end import KEY_END_STATE, VAL_END_STATE_NONE, VAL_END_STATE_RECVD, End
from flame.selector import AbstractSelector, SelectorReturnType

logger = logging.getLogger(__name__)

SEND_TIMEOUT_WAIT_S = 90  # 90 seconds timeout

PROP_END_ID = "end_id"
PROP_STAT_UTILITY = "stat_utility"
PROP_AVL_STATE = "avl_state"


class AsyncRandomSelector(AbstractSelector):
    """A AsyncFL selector class based on Oort."""

    def __init__(self, **kwargs):
        """Initailize instance."""
        super().__init__(**kwargs)

        ml_framework_in_use = get_ml_framework_in_use()
        if ml_framework_in_use != MLFramework.PYTORCH:
            raise NotImplementedError(
                "FedBalancer is currently only implemented in PyTorch;"
            )

        self.round = 0
        self.is_async = True

        try:
            self.c = kwargs["c"]
        except KeyError:
            raise KeyError("c (concurrency level) is not specified in config")

        try:
            self.agg_goal = kwargs["aggGoal"]
        except KeyError:
            raise KeyError("aggGoal is not specified in config")


        if self.agg_goal < 0:
            self.agg_goal = 1

        # #### CHANGES BASED OFF FEDBUFF FOR ASYNCFL
        # Tracking selected ends to ensure selection correctness for
        # each round (a trainer can participate only once per round).
        self.all_selected = dict()
        self.selected_ends = dict()

        # Tracks weight updates received from trainers and makes them
        # available to select again
        self.ordered_updates_recv_ends = list()

        # Tracks timeouted trainers and number of times it happened to
        # a trainer
        self.track_trainer_timeouts = dict()

        # Tracks trainers that were selected but left training in
        # between
        self.track_selected_trainers_which_left = dict()
        self.check_three_state_avl = True  # kept for backward compat; superseded by _task_eligible_states

        # Configurable task → eligible avl_state mapping.
        # Default matches current hardcoded behavior.
        # Override via selector.kwargs["task_eligible_states"] in aggregator.json.
        _default_eligible_states = {
            "train": [TrainerAvailState.AVL_TRAIN.value],
            "eval": [
                TrainerAvailState.AVL_EVAL.value,
                TrainerAvailState.AVL_TRAIN.value,
            ],
        }
        raw_eligible = kwargs.get("task_eligible_states", _default_eligible_states)
        _valid_states = {v.value for v in TrainerAvailState}
        for task_name, states in raw_eligible.items():
            for s in states:
                if s not in _valid_states:
                    raise ValueError(
                        f"task_eligible_states['{task_name}'] contains unknown state "
                        f"'{s}'. Valid states: {sorted(_valid_states)}"
                    )
        self._task_eligible_states: dict = raw_eligible
        logger.info(
            f"[TaskEligibility] task_eligible_states = {self._task_eligible_states}"
        )

        # Track sliding window statistics for the selector
        self._selector_stats = {}
        for task in ["train", "eval"]:
            self._selector_stats[task] = {"data": {}, "summary": {}}
            for metric in ["util", "speed", "round"]:
                for window in [50, 100, 200]:
                    key = f"{metric}_last_{window}"
                    self._selector_stats[task]["data"][key] = deque(maxlen=window)

        self._select_run_counter = 0

    def select(
        self,
        ends: dict[str, End],
        channel_props: dict[str, Scalar],
        trainer_unavail_list: list,
        task_to_perform: str = "train",
        **kwargs,
    ) -> SelectorReturnType:
        """Return k number of ends from the given ends.

        NOTE: It incorporates the same send/recv mechanism from
        fedbuff. [From fedbuff selector]: Select ends from the given
        ends to meet concurrency level. This select method chooses
        ends differently depending on what state a channel is in. In
        'send' state, it chooses ends that are not in
        self.selected_ends. In 'recv' state, it chooses all ends from
        self.selected_ends. Essentially, if an end is in
        self.selected_ends, it means that we sent some message already
        to that end. For such an end, we exclude it from send and
        include it for recv in return.
        """
        logger.info("calling async random select")
        # Extract aggregator version and trainer version states for staleness tracking
        agg_version_state = kwargs.get("agg_version_state")
        trainer_version_states = kwargs.get("trainer_version_states")
        logger.debug(
            f"Aggregator version state (model_version, data_id, iteration_id): {agg_version_state}"
        )
        logger.debug(f"Trainer version states: {trainer_version_states}")

        if task_to_perform == "train":
            concurrency = min(len(ends), self.c)

        logger.info(
            f"Task: {task_to_perform}, len(ends): {len(ends)}, c: {self.c}, chosen concurrency: {concurrency}"
        )

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

        if channel_props[KEY_CH_STATE] == VAL_CH_STATE_SEND:
            logger.debug(
                f"Inside send state: aggregator version state (model_version, data_id, iteration_id): {agg_version_state}"
            )
            logger.debug(
                f"Inside send state: trainer version states: {trainer_version_states}"
            )
            results = self._handle_send_state(
                ends=eligible_ends,
                concurrency=concurrency,
                channel_props=channel_props,
                trainer_unavail_list=trainer_unavail_list,
                task_to_perform=task_to_perform,
                agg_version_state=agg_version_state,
                trainer_version_states=trainer_version_states,
            )

            if len(results) is not 0:
                self._select_run_counter += 1


        elif channel_props[KEY_CH_STATE] == VAL_CH_STATE_RECV:
            results = self._handle_recv_state(ends, concurrency)

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

    def select_random(self, ends: dict[str, End], num_of_ends: int) -> dict[str, None]:
        """Randomly select num_of_ends ends."""

        selected_random_ends = set(random.sample(list(ends), num_of_ends))
        logger.debug(f"selected_random_ends: {selected_random_ends}")

        return {key: None for key in selected_random_ends}

    def _cleanup_provided_ends(
        self, ends_to_cleanup: dict[str, End], ends: dict[str, End]
    ):
        """Clean-up a specific end so it becomes eligible for sampling again - reject stale updates in FwdLLM (async)"""

        selected_ends = self.selected_ends.get(self.requester, set())
        for end_id, _ in ends_to_cleanup.items():
            state = ends[end_id].get_property(KEY_END_STATE)
            logger.info(f"Cleaning end {end_id}, current state: {state}")

            # reset only if it's in received state
            if state == VAL_END_STATE_RECVD:
                ends[end_id].set_property(KEY_END_STATE, VAL_END_STATE_NONE)
                logger.debug(
                    f"Setting {end_id} state to {VAL_END_STATE_NONE}, "
                    f"and"
                    f" removing from selected_ends and all_selected"
                )

            # remove from active selection tracking
            if end_id in selected_ends:
                selected_ends.remove(end_id)
                logger.debug(f"Removed {end_id} from selected_ends")

            if end_id in self.all_selected:
                del self.all_selected[end_id]
                logger.debug(f"Removed {end_id} from all_selected")

        # update the mapping back
        self.selected_ends[self.requester] = selected_ends
        logger.info(
            f"Cleanup complete. Freed [{ends}] end(s) for resampling; state set to {VAL_END_STATE_NONE}."
        )

    # #### CHANGES BASED OFF FEDBUFF FOR ASYNCFL
    def _cleanup_recvd_ends(self, ends: dict[str, End]):
        """Clean up ends whose a message was received, from selected
        ends.

        Note: It sets the end state to none which makes it eligible to
        be sampled again. This can cause problems if sampled in the
        same round. Thus, for aggregator, the _cleanup_recvd_ends
        should be triggered only after aggregation of weights succeeds
        on meeting agg_goal."""
        logger.debug(
            f"clean up recvd ends. selected_ends: {self.selected_ends}, ends: {ends.keys()}"
        )

        selected_ends = self.selected_ends[self.requester]
        logger.debug(
            f"self.requester: {self.requester} and selected_ends: "
            f"{selected_ends} before processing"
        )

        num_ends_to_remove = min(len(self.ordered_updates_recv_ends), self.agg_goal)
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

            logger.debug(
                f"Ends to remove based on trainer updates received: {ends_to_remove}"
            )

            logger.debug(f"All ends to remove (train ): {ends_to_remove}")

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
                            f"and"
                            f" removing from selected_ends and all_selected"
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
                        logger.debug(
                            f"Found end {end_id} in state {VAL_END_STATE_NONE}. Might have "
                            f"left/rejoined. Need to remove it from "
                            f"selected_ends and self.all_selected "
                            f"if it was selected"
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
            logger.warning(
                f"End_id {end_id} remove check from all_selected failed. "
                f"Need to check"
            )

    def _handle_send_state(
        self,
        ends: dict[str, End],
        concurrency: int,
        channel_props: dict[str, Scalar],
        trainer_unavail_list: list = None,
        task_to_perform: str = "train",
        agg_version_state=None,  # (model_version, data_id, iteration_id)
        trainer_version_states: dict[str, tuple[int, int, int]] = None,
    ) -> SelectorReturnType:
        selected_ends = self.selected_ends[self.requester]
        logger.debug(f"Inside handle send state: aggregator version state {agg_version_state}")
        logger.debug(
            f"Inside handle send state: trainer version states {trainer_version_states}"
        )
        # Check for invalid selections and remove them
        for end_id in list(selected_ends):
            if end_id not in ends:
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

        logger.debug(f"Current selected_ends: {selected_ends}")

        extra = max(0, concurrency - len(selected_ends))

        logger.debug(
            f"c: {concurrency}, "
            f"len(selected_ends): {len(selected_ends)}, extra: {extra}, selected_ends: {selected_ends},"
            f"len(ends): {len(ends)}"
        )
        candidates = []

        if extra == 0:
            logger.debug(f"extra: {extra}, nothing to select")
            return {}

        round = channel_props["round"] if "round" in channel_props else 0
        logger.debug(f"let's select {extra} ends for round {round}")


        # Invalidate previous all_selected entry if you don't get an
        # update in UPDATE_TIMEOUT_WAIT_S. The client might have
        # dropped the message with transient unavailability.

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
                        f"got its update in {SEND_TIMEOUT_WAIT_S}. "
                        f"Last weight send timestamp was: {trainer_weight_send_timestamp_s}"
                    )

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

                    logger.debug(
                        f"Timeout for trainer: {end} with count "
                        f"{self.track_trainer_timeouts[end]}. "
                        f"num_of_timeouts_occured : "
                        f"{num_of_timeouts_occured}, "
                        f"total_time_spent_timeouts_s: "
                        f"{total_time_spent_timeouts_s}, "
                        f"Timeout frequency: {self.track_trainer_timeouts}"
                    )

                    # delete the end from self.all_selected
                    if end in self.all_selected.keys():
                        del self.all_selected[end]

        # TODO: (DG) Add code to allow only those ends (not in
        # all_selected) to be passed. filtered_ends consists of ends
        # that are not in all_selected and can be picked in this round
        # i.e. avoids repeating a trainer in the same round
        filtered_ends = dict()

        # track the ends that are eligible vs ineligible based on
        # their state
        count_avl_train = 0
        count_ineligible = 0

        # Check the eligible set first. Out of the ends, how many are
        # not in all_selected? Only those are eligible since the rest
        # have weights already sent to them for either train/eval
        # task.
        count_eligible_set_to_check = [
            end for end in ends if end not in self.all_selected
        ]
        logger.debug(
            f"Before creating filtered_ends. count_eligible_set_to_check: {len(count_eligible_set_to_check)} from total {len(ends)} ends."
        )

        for end_id in ends:
            if end_id not in self.all_selected.keys():
                logger.debug(
                    f"Creating filtered ends. Checking end id {end_id}, avl_state = {ends[end_id].get_property(PROP_AVL_STATE)}"
                )

                # If check_three_state_avl=False, no more checks,
                # directly add end to filtered_ends

                # If check_three_state_avl=True, filtered ends needs
                # to be populated based on the following conditions:
                # For task_to_perform=train, eligible ends are in
                # states {avl_train, None} For task_to_perform=eval,
                # eligible ends are in states {avl_train, avl_eval
                # None}

                curr_end_id_avl_state = ends[end_id].get_property(PROP_AVL_STATE)
                eligible_states_for_task = self._task_eligible_states.get(
                    task_to_perform, []
                )
                # None avl_state means no heartbeat state set — always eligible.
                state_eligible = (
                    curr_end_id_avl_state is None
                    or curr_end_id_avl_state in eligible_states_for_task
                )

                if state_eligible:
                    filtered_ends[end_id] = ends[end_id]
                    count_avl_train += 1
                    logger.debug(
                        f"Adding end {end_id} to filtered_ends: "
                        f"task={task_to_perform}, avl_state={curr_end_id_avl_state}, "
                        f"eligible_states={eligible_states_for_task}"
                    )
                else:
                    count_ineligible += 1
                    logger.debug(
                        f"Skipping end {end_id}: task={task_to_perform}, "
                        f"avl_state={curr_end_id_avl_state} not eligible "
                        f"(eligible_states={eligible_states_for_task})"
                    )

        logger.info(
            f"Filtered ends created. count_avl_train: {count_avl_train},  count_ineligible: {count_ineligible}"
        )

        if agg_version_state is not None and trainer_version_states is not None:
            curr_model_version, curr_data_id, curr_iteration_id = agg_version_state
            logger.info(f"Trainer version states: {trainer_version_states}")
            logger.info(f"Handle send state: aggregator version state {agg_version_state}")
            # Filter out trainers who already received this same triplet
            eligible_filtered_ends = {}
            logger.debug(f"Filtered ends: {filtered_ends.items()}")
            for end_id, end in filtered_ends.items():
                prev_state = trainer_version_states.get(end_id)
                logger.debug(f"Prev version state: {prev_state}")

                if prev_state != agg_version_state:
                    logger.debug(f"Not skipping trainer: {end_id}")
                    eligible_filtered_ends[end_id] = end
                else:
                    logger.info(
                        f"Skipping trainer: {end_id} already has same "
                        f"(model_version={curr_model_version}, "
                        f"iteration_id={curr_iteration_id}, data_id={curr_data_id})"
                    )
            filtered_ends = eligible_filtered_ends

        # extra informs about maximum possible available ends that can
        # be picked to meet the concurrency target. But it might count
        # infeasible ends too (ends that have already particpated in
        # the round). It is essentially a superset of feasible and
        # infeasible. Maximum feasible comes from filtered_ends. We
        # define and henceforth use feasible_extra to (i) use extra's
        # knowledge of how many to pick and (ii) use filtered_ends
        # knowledge of what is feasible to pick Eg scenarios:
        # (extra=1, filtered=3),  (extra=2, filtered=2), (extra=3,
        # filtered=1)
        feasible_extra = min(extra, len(filtered_ends))
        logger.info(
            f"desired extra: {extra}, len(filtered_ends): {len(filtered_ends)}, feasible_extra: {feasible_extra}"
        )

        # Early exit if filtered_ends is none (can happen when all
        # ends available are less than concurrency requirement)
        if len(filtered_ends) == 0:
            logger.info(
                f"len(filtered_ends): {len(filtered_ends)}, hence returning "
                f"with empty candidates"
            )
            return {}

        # This is only for train as of now, can be extended for eval in the future
        if task_to_perform == "train":
            # Make a filter of blocklist ends
            # blocklist_end_ids = self.find_blocklists(filtered_ends)

            if trainer_unavail_list != []:
                logger.info(
                    "### Oort select got non-empty trainer_unavail_list, will "
                    "remove unavail trainers from round"
                )

            self.round = round

            logger.info(
                f"Round: {self.round}, will sample feasible_extra: "
                f"{feasible_extra} from len(filtered_ends): "
                f"{len(filtered_ends)}"
            )
            candidates_dict = self.select_random(
                filtered_ends, num_of_ends=feasible_extra
            )
                # Invoke process_chosen_candidate_dict(). It will
                # appropriately add candidates to selected_ends and
                # all_selected
            self.process_chosen_candidate_dict(
                candidates_dict=candidates_dict, selected_ends=selected_ends
            )

            logger.info(
                f"handle_send_state returning "
                f"candidates_dict: {candidates_dict}"
            )

            return candidates_dict


    def _handle_recv_state(
        self, ends: dict[str, End], concurrency: int
    ) -> SelectorReturnType:
        selected_ends = self.selected_ends[self.requester]

        # from the selected ends, remove those that are in recv state
        # already This is done to avoid waiting on trainers that you
        # have already heard from. If selected ends is empty, get()
        # will proceed and wait on distribute_weights before running
        # again. Thus, it avoids stalling and ensures progress
        for end_id in list(selected_ends):
            # trainer might have become unavailable, check if it is
            # still available first
            if end_id in ends:
                curr_end_state = ends[end_id].get_property(KEY_END_STATE)
                if curr_end_state == VAL_END_STATE_RECVD:
                    selected_ends.remove(end_id)
                    logger.debug(
                        f"Removed end_id {end_id} from selected ends since it "
                        f"was already in {curr_end_state} state"
                    )
            else:

                logger.debug(
                    f"Tried to check state of end {end_id} but it is no "
                    f"longer in self._ends"
                )

        if len(selected_ends) == 0:
            logger.debug(f"len(selected_ends)=0, let's select {concurrency} ends")

            candidates = dict()
            for end_id, end in ends.items():
                curr_end_state = end.get_property(KEY_END_STATE)
                # candidates[end_id] = end
                if end_id not in self.all_selected.keys():
                    if curr_end_state != VAL_END_STATE_NONE:
                        logging.info(
                            f"end_id {end_id} not in all_selected and in state: {curr_end_state}, adding "
                            f"to candidates: key {end_id}, val: {end}"
                        )
                        candidates[end_id] = end
                    else:
                        logging.debug(
                            f"end_id {end_id} not in all_selected but in state: {curr_end_state}, not adding "
                            f"to candidates"
                        )

            cc = min(len(candidates), concurrency)
            logger.debug(
                f"Will pick cc: {cc} as min(candidates,concurrency) "
                f"from candidates: {candidates}"
            )
            selected_ends = set(random.sample(list(candidates), cc))

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

    def process_chosen_candidate_dict(
        self,
        candidates_dict: dict[str, None],
        selected_ends: set[str],
    ):
        candidates = list(candidates_dict.keys())
        logger.debug(
            f"Got candidates_dict as {candidates_dict} after " f"select_random"
        )
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
            f"self.all_selected {self.all_selected} after combining"
            f" with candidates {candidates}"
        )

        logger.debug("finished processing candidates_dict")
