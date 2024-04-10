# Copyright 2024 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import logging
import random

from flame.channel_manager import ChannelManager
from flame.config import Config
from flame.mode.composer import Composer
from flame.mode.message import MessageType
from flame.mode.role import Role
from flame.mode.tasklet import Loop, Tasklet

logger = logging.getLogger(__name__)

TAG_COORDINATE_WITH_TOP_AGG = "coordinateWithTopAgg"
TAG_COORDINATE_WITH_MID_AGG = "coordinateWithMidAgg"
TAG_COORDINATE_WITH_LEAF_AGG = "coordinateWithLeafAgg"
TAG_COORDINATE_WITH_TRAINER = "coordinateWithTrainer"


class Coordinator(Role):
    def __init__(self, config: Config, **kwargs):
        self.config = config

        self.channel_manager = self.build_channel_manager()

        self._rounds = self.config.hyperparameters.rounds
        self._round = 1
        self._work_done = False

        self.leaf_agg_to_trainer = dict()
        self.trainer_to_leaf_agg = dict()

        self.mid_agg_to_leaf_agg = dict()
        self.leaf_agg_to_mid_agg = dict()

    def build_channel_manager(self):
        channel_manager = ChannelManager()
        channel_manager(self.config)
        channel_manager.join_all()

        return channel_manager

    def get_channel(self, tag: str):
        """Return channel of a given tag when it is ready to use."""
        channel = self.channel_manager.get_by_tag(tag)
        if not channel:
            raise ValueError(f"channel not found for tag {tag}")

        channel.await_join()

        return channel

    def await_mid_leaf_aggs_and_trainers(self):
        """Wait for mid and leaf aggregators and trainers to join."""
        logger.info("waiting for mid and leaf aggs and trainers")

        trainer_channel = self.get_channel(TAG_COORDINATE_WITH_TRAINER)
        leaf_aggregator_channel = self.get_channel(TAG_COORDINATE_WITH_LEAF_AGG)
        mid_aggregator_channel = self.get_channel(TAG_COORDINATE_WITH_MID_AGG)

        # set necessary properties to help channel decide how to select ends
        trainer_channel.set_property("round", self._round)
        leaf_aggregator_channel.set_property("round", self._round)
        mid_aggregator_channel.set_property("round", self._round)

        logger.info("all mid, leaf aggs and trainers joined")

    def _round_robin_pair(self, lower_ends, upper_ends, lower_to_upper, upper_to_lower):
        """round-robin pair scheme."""
        lower_idx = 0
        for lower_end in lower_ends:
            upper_end = upper_ends[lower_idx % len(upper_ends)]

            lower_to_upper[lower_end] = upper_end
            upper_to_lower[upper_end].append(lower_end)
            lower_idx += 1

    def _best_fit_pair(self, trainer_ends, agg_ends):
        """best-fit pair aggregator with trainers."""
        pass

    def _worst_fit_pair(self, trainer_ends, agg_ends):
        """worst-fit pair aggregator with trainers."""
        pass

    def _first_fit_pair(self, trainer_ends, agg_ends):
        """first-fit pair aggregator with trainers."""
        pass

    def _count_assigned_trainers_to_agg(self):
        """Count assigned trainers to aggregator. (for debugging)"""
        for agg_end, trainers in self.agg_to_trainer.items():
            logger.debug(f"agg_end: {agg_end}, # of trainers: {len(trainers)}")

    def _get_good_ends(self, channel, agg_ends):
        for agg_end in agg_ends:
            logger.debug(f"sending meta info req to {agg_end}")
            channel.send(agg_end, {MessageType.META_INFO_REQ: ""})

        bad_ends = set()
        for msg, metadata in channel.recv_fifo(agg_ends):
            end, _ = metadata
            if not msg or MessageType.META_INFO_RES not in msg:
                bad_ends.add(end)
                logger.debug(f"No meta info response from {end}; skipping it")
                continue

            # meta info can be used to mapping leaf aggregators to trainers
            # TODO: implement necessary logic
            logger.info(f"got {msg[MessageType.META_INFO_RES]} from {end}")

        good_ends = [end for end in agg_ends if end not in bad_ends]

        return good_ends

    def pair_leaf_aggs_and_trainers(self):
        """Pair leaf aggregators with trainers."""
        logger.debug("paring leaf aggs and trainers")
        trainer_channel = self.get_channel(TAG_COORDINATE_WITH_TRAINER)
        aggregator_channel = self.get_channel(TAG_COORDINATE_WITH_LEAF_AGG)

        trainer_ends = trainer_channel.ends()
        agg_ends = aggregator_channel.ends()

        trainer_ends = self._get_good_ends(trainer_channel, trainer_ends)
        logger.debug(f"good trainer ends: {trainer_ends}")

        agg_ends = self._get_good_ends(aggregator_channel, agg_ends)
        logger.debug(f"good leaf agg ends: {agg_ends}")

        # reset
        self.leaf_agg_to_trainer = dict()
        self.trainer_to_leaf_agg = dict()

        # initialize leaf_agg_to_trainer
        for agg_end in agg_ends:
            self.leaf_agg_to_trainer[agg_end] = list()

        # TODO: making pairing scheme configurable
        self._round_robin_pair(trainer_ends, agg_ends, self.trainer_to_leaf_agg, self.leaf_agg_to_trainer)

        logger.debug(f"trainer to leaf agg mapping: {self.trainer_to_leaf_agg}")
        logger.debug(f"leaf agg to trainer mapping: {self.leaf_agg_to_trainer}")

        logger.debug("finished pairing leaf aggs and trainers")

    def pair_mid_aggs_and_leaf_aggs(self):
        """Pair mid aggregators with leaf aggregators."""
        logger.debug("paring mid aggs and leaf aggs")
        leaf_aggregator_channel = self.get_channel(TAG_COORDINATE_WITH_LEAF_AGG)
        mid_aggregator_channel = self.get_channel(TAG_COORDINATE_WITH_MID_AGG)

        leaf_agg_ends = leaf_aggregator_channel.ends()
        mid_agg_ends = mid_aggregator_channel.ends()

        # leaf_agg_ends = self._get_good_ends(leaf_aggregator_channel, leaf_agg_ends)
        # logger.debug(f"good leaf agg ends: {leaf_agg_ends}")

        mid_agg_ends = self._get_good_ends(mid_aggregator_channel, mid_agg_ends)
        logger.debug(f"good mid agg ends: {mid_agg_ends}")

        # reset
        self.mid_agg_to_leaf_agg = dict()
        self.leaf_agg_to_mid_agg = dict()

        # initialize mid_agg_to_leaf_agg
        for mid_agg_end in mid_agg_ends:
            self.mid_agg_to_leaf_agg[mid_agg_end] = list()

        # filter out leaf_agg_end that has no assigned trainers
        leaf_aggs_w_trainer = list()
        for leaf_agg, trainers in self.leaf_agg_to_trainer.items():
            if len(trainers) == 0:
                logger.debug(f"no trainer assigned for leaf agg {leaf_agg}")
                continue

            leaf_aggs_w_trainer.append(leaf_agg)
        
        leaf_agg_ends = [agg for agg in leaf_agg_ends if agg in leaf_aggs_w_trainer]

        # TODO: making pairing scheme configurable
        self._round_robin_pair(leaf_agg_ends, mid_agg_ends, self.leaf_agg_to_mid_agg, self.mid_agg_to_leaf_agg)

        logger.debug(f"mid agg to leaf agg mapping: {self.mid_agg_to_leaf_agg}")
        logger.debug(f"leaf agg to mid agg mapping: {self.leaf_agg_to_mid_agg}")

        logger.debug("finished pairing mid aggs and leaf aggs")

    def send_selected_middle_aggregators(self):
        """Send selected middle aggregator list to top aggregator."""
        logger.debug("sending selected mid aggs to top agg")
        top_agg_channel = self.get_channel(TAG_COORDINATE_WITH_TOP_AGG)

        mid_aggs = list()
        for mid_agg, leaf_aggs in self.mid_agg_to_leaf_agg.items():
            if len(leaf_aggs) == 0:
                logger.debug(f"no leaf agg assigned for mid agg {mid_agg}")
                continue

            mid_aggs.append(mid_agg)

        msg = {MessageType.COORDINATED_ENDS: mid_aggs, MessageType.EOT: self._work_done}
        end = top_agg_channel.one_end()
        top_agg_channel.send(end, msg)

        logger.debug("finished sending selected mid aggs to top agg")

    def send_selected_leaf_aggregators(self):
        """Send selected leaf aggregator list to middle aggregator."""
        logger.debug("sending selected leaf aggs to mid agg")
        mid_agg_channel = self.get_channel(TAG_COORDINATE_WITH_MID_AGG)

        for mid_agg, leaf_aggs in self.mid_agg_to_leaf_agg.items():
            msg = {
                MessageType.COORDINATED_ENDS: leaf_aggs,
                MessageType.EOT: self._work_done,
            }
            mid_agg_channel.send(mid_agg, msg)

        logger.debug("finished sending selected leaf aggs to mid agg")

    def send_selected_trainers(self):
        """Send selected trainer list to leaf aggregator."""
        logger.debug("calling send_selected_trainers()")
        leaf_agg_channel = self.get_channel(TAG_COORDINATE_WITH_LEAF_AGG)

        for leaf_agg, trainers in self.leaf_agg_to_trainer.items():
            msg = {
                MessageType.COORDINATED_ENDS: trainers,
                MessageType.EOT: self._work_done,
            }
            leaf_agg_channel.send(leaf_agg, msg)
        logger.debug("exited send_selected_trainers()")

    def send_selected_leaf_aggregator(self):
        """Send selected leaf aggregator ID to each trainer."""
        logger.debug("calling send_selected_leaf_aggregator()")
        trainer_channel = self.get_channel(TAG_COORDINATE_WITH_TRAINER)

        for trainer, leaf_agg in self.trainer_to_leaf_agg.items():
            msg = {MessageType.COORDINATED_ENDS: leaf_agg, MessageType.EOT: self._work_done}
            trainer_channel.send(trainer, msg)
        logger.debug("exited send_selected_leaf_aggregator()")

    def send_selected_middle_aggregator(self):
        """Send selected middle aggregator ID to each leaf aggregator."""
        logger.debug("calling send_selected_middle_aggregator()")
        leaf_agg_channel = self.get_channel(TAG_COORDINATE_WITH_LEAF_AGG)

        for leaf_agg, mid_agg in self.leaf_agg_to_mid_agg.items():
            msg = {MessageType.COORDINATED_ENDS: mid_agg, MessageType.EOT: self._work_done}
            leaf_agg_channel.send(leaf_agg, msg)
        logger.debug("exited send_selected_middle_aggregator()")

    def increment_round(self) -> None:
        """Increment the round counter."""
        logger.debug(f"incrementing current round: {self._round}")
        logger.debug(f"total rounds: {self._rounds}")
        self._round += 1
        self._work_done = self._round > self._rounds

        logger.debug(f"incremented round to {self._round}")

    def inform_end_of_training(self) -> None:
        """Inform the end of training."""
        top_agg_channel = self.get_channel(TAG_COORDINATE_WITH_TOP_AGG)
        top_agg_channel.broadcast({MessageType.EOT: self._work_done})

        mid_agg_channel = self.get_channel(TAG_COORDINATE_WITH_MID_AGG)
        mid_agg_channel.broadcast({MessageType.EOT: self._work_done})

        leaf_agg_channel = self.get_channel(TAG_COORDINATE_WITH_LEAF_AGG)
        leaf_agg_channel.broadcast({MessageType.EOT: self._work_done})

        trainer_channel = self.get_channel(TAG_COORDINATE_WITH_TRAINER)
        trainer_channel.broadcast({MessageType.EOT: self._work_done})

        logger.debug("done broadcasting end-of-training")

    ###
    # Functions  in the following are defined as abstraction functions in Role
    # But they are irrelevant to coordinator; so, we simply bypass them
    # TODO: revisit what functions need to be defined as abstract method in Role

    def internal_init(self) -> None:
        pass

    def get(self, tag: str) -> None:
        # NOTE: To simplify implementation, bypass this function
        # TODO: this is abstract function in Role;
        #       need to revisit if this function needs to be defined there
        pass

    def put(self, tag: str) -> None:
        # NOTE: To simplify implementation, bypass this function
        # TODO: this is abstract function in Role;
        #       need to revisit if this function needs to be defined there
        pass

    def initialize(self) -> None:
        pass

    def load_data(self) -> None:
        pass

    def train(self) -> None:
        pass

    def evaluate(self) -> None:
        pass

    ###

    def compose(self) -> None:
        """Compose role with tasklets."""
        with Composer() as composer:
            self.composer = composer

            task_await = Tasklet(
                "await_mid_leaf_aggs_and_trainers", self.await_mid_leaf_aggs_and_trainers
            )

            task_pairing_leaf_aggs_and_trainers = Tasklet(
                "pair_leaf_aggs_and_trainers", self.pair_leaf_aggs_and_trainers
            )

            task_pairing_mid_aggs_and_leaf_aggs = Tasklet(
                "pair_mid_aggs_and_leaf_aggs", self.pair_mid_aggs_and_leaf_aggs
            )

            task_send_mid_aggs_to_top_agg = Tasklet(
                "send_selected_middle_aggregators",
                self.send_selected_middle_aggregators,
            )

            task_send_leaf_aggs_to_mid_agg = Tasklet(
                "send_selected_leaf_aggregators",
                self.send_selected_leaf_aggregators,
            )

            task_send_trainers_to_leaf_agg = Tasklet(
                "send_selected_trainers", self.send_selected_trainers
            )

            task_send_mid_agg_to_leaf_agg = Tasklet(
                "send_selected_middle_aggregator", self.send_selected_middle_aggregator
            )

            task_send_leaf_agg_to_trainer = Tasklet(
                "send_selected_leaf_aggregator", self.send_selected_leaf_aggregator
            )

            task_increment_round = Tasklet("inc_round", self.increment_round)

            task_inform_eot = Tasklet(
                "inform_end_of_training", self.inform_end_of_training
            )

        loop = Loop(loop_check_fn=lambda: self._work_done)
        (
            loop(
                task_await
                >> task_pairing_leaf_aggs_and_trainers
                >> task_pairing_mid_aggs_and_leaf_aggs
                >> task_send_mid_aggs_to_top_agg
                >> task_send_leaf_aggs_to_mid_agg
                >> task_send_trainers_to_leaf_agg
                >> task_send_mid_agg_to_leaf_agg
                >> task_send_leaf_agg_to_trainer
                >> task_increment_round
            )
            >> task_inform_eot
        )

    def run(self) -> None:
        """Run role."""
        self.composer.run()

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the top level aggregator role."""
        return [
            TAG_COORDINATE_WITH_TOP_AGG,
            TAG_COORDINATE_WITH_MID_AGG,
            TAG_COORDINATE_WITH_LEAF_AGG,
            TAG_COORDINATE_WITH_TRAINER,
        ]
