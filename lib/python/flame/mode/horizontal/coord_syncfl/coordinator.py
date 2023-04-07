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
TAG_COORDINATE_WITH_TRAINER = "coordinateWithTrainer"


class Coordinator(Role):
    def __init__(self, config: Config, **kwargs):
        self.config = config

        self.channel_manager = self.build_channel_manager()

        self._rounds = self.config.hyperparameters.rounds
        self._round = 1
        self._work_done = False

        self.agg_to_trainer = dict()
        self.trainer_to_agg = dict()

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

    def await_mid_aggs_and_trainers(self):
        """Wait for middle aggregators and trainers to join."""
        logger.info("waiting for mid aggs and trainers")

        trainer_channel = self.get_channel(TAG_COORDINATE_WITH_TRAINER)
        aggregator_channel = self.get_channel(TAG_COORDINATE_WITH_MID_AGG)

        # set necessary properties to help channel decide how to select ends
        trainer_channel.set_property("round", self._round)
        aggregator_channel.set_property("round", self._round)

        logger.info("both mid aggs and trainers joined")

    def pair_mid_aggs_and_trainers(self):
        """Pair middle aggregators with trainers."""
        logger.debug("paring mid aggs and trainers")
        trainer_channel = self.get_channel(TAG_COORDINATE_WITH_TRAINER)
        aggregator_channel = self.get_channel(TAG_COORDINATE_WITH_MID_AGG)

        trainer_ends = trainer_channel.ends()
        agg_ends = aggregator_channel.ends()

        # send meta information request
        for agg_end in agg_ends:
            logger.debug(f"sending meta info req to {agg_end}")
            aggregator_channel.send(agg_end, {MessageType.META_INFO_REQ: ""})

        bad_ends = set()
        for msg, metadata in aggregator_channel.recv_fifo(agg_ends):
            end, _ = metadata
            if not msg or MessageType.META_INFO_RES not in msg:
                bad_ends.add(end)
                logger.debug(f"No meta info response from {end}; skipping it")
                continue

            # meta info can be used to mapping middle aggregators to trainers
            # TODO: implement necessary logic
            logger.info(f"got {msg[MessageType.META_INFO_RES]} from {end}")

        agg_ends = [end for end in agg_ends if end not in bad_ends]

        logger.debug(f"good mid agg ends: {agg_ends}")

        # reset
        self.agg_to_trainer = dict()
        self.trainer_to_agg = dict()

        # initialize agg_to_trainer
        for agg_end in agg_ends:
            self.agg_to_trainer[agg_end] = list()

        # randomly pair aggregator with trainers
        # TODO: other pairing can be implemented
        for trainer_end in trainer_ends:
            agg_end = random.choice(agg_ends)
            self.trainer_to_agg[trainer_end] = agg_end

            self.agg_to_trainer[agg_end].append(trainer_end)

        logger.debug("finished paring mid aggs and trainers")

    def send_selected_middle_aggregators(self):
        """Send selected middle aggregator list to top aggregator."""
        logger.debug("sending selected mid aggs to top agg")
        top_agg_channel = self.get_channel(TAG_COORDINATE_WITH_TOP_AGG)

        mid_aggs = list()
        for agg, trainers in self.agg_to_trainer.items():
            if len(trainers) == 0:
                logger.debug(f"no trainer assigned for mid agg {agg}")
                continue

            mid_aggs.append(agg)

        msg = {MessageType.COORDINATED_ENDS: mid_aggs, MessageType.EOT: self._work_done}
        end = top_agg_channel.one_end()
        top_agg_channel.send(end, msg)

        logger.debug("finished sending selected mid aggs to top agg")

    def send_selected_trainers(self):
        """Send selected trainer list to middle aggregator."""
        logger.debug("calling send_selected_trainers()")
        mid_agg_channel = self.get_channel(TAG_COORDINATE_WITH_MID_AGG)

        for agg, trainers in self.agg_to_trainer.items():
            msg = {
                MessageType.COORDINATED_ENDS: trainers,
                MessageType.EOT: self._work_done,
            }
            mid_agg_channel.send(agg, msg)
        logger.debug("exited send_selected_trainers()")

    def send_selected_middle_aggregator(self):
        """Send selected middle aggregator ID to each trainer."""
        logger.debug("calling send_selected_middle_aggregator()")
        trainer_channel = self.get_channel(TAG_COORDINATE_WITH_TRAINER)

        for trainer, agg in self.trainer_to_agg.items():
            msg = {MessageType.COORDINATED_ENDS: agg, MessageType.EOT: self._work_done}
            trainer_channel.send(trainer, msg)
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

            task_await = Tasklet("", self.await_mid_aggs_and_trainers)

            task_pairing = Tasklet("", self.pair_mid_aggs_and_trainers)

            task_send_mid_aggs_to_top_agg = Tasklet(
                "", self.send_selected_middle_aggregators
            )

            task_send_trainers_to_agg = Tasklet("", self.send_selected_trainers)

            task_send_agg_to_trainer = Tasklet("", self.send_selected_middle_aggregator)

            task_increment_round = Tasklet("", self.increment_round)

            task_inform_eot = Tasklet("", self.inform_end_of_training)

        loop = Loop(loop_check_fn=lambda: self._work_done)
        (
            loop(
                task_await
                >> task_pairing
                >> task_send_mid_aggs_to_top_agg
                >> task_send_trainers_to_agg
                >> task_send_agg_to_trainer
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
            TAG_COORDINATE_WITH_TRAINER,
        ]
