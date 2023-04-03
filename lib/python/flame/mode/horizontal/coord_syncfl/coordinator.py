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

TAG_SELECT_TRAINERS = "selectTrainers"
TAG_SELECT_AGGREGATOR = "selectAggregator"
TAG_CHECK_EOT = "checkEOT"


class Coordinator(Role):
    def __init__(self, config: Config, **kwargs):
        self.config = config

        self.channel_manager = self.build_channel_manager()

        self._work_done = False

        self._round = 0

        self.agg_to_trainer = dict()
        self.trainer_to_agg = dict()

    def build_channel_manager(self):
        channel_manager = ChannelManager()
        channel_manager(self.config)
        channel_manager.join_all()

        return channel_manager

    def check_eot(self):
        logger.info("calling check_eot")

        top_agg_channel = self.channel_manager.get_by_tag(TAG_CHECK_EOT)
        if not top_agg_channel:
            raise ValueError("top aggregator channel not found")

        top_agg_channel.await_join()
        end = top_agg_channel.one_end()
        msg, _ = top_agg_channel.recv(end)

        if MessageType.EOT in msg:
            self._work_done = msg[MessageType.EOT]

        logger.info(f"work_done: {self._work_done}")

        self._round += 1

    def await_mid_aggs_and_trainers(self):
        logger.info("waiting for mid aggs and trainers")
        trainer_channel = self.channel_manager.get_by_tag(TAG_SELECT_AGGREGATOR)
        aggregator_channel = self.channel_manager.get_by_tag(TAG_SELECT_TRAINERS)

        trainer_channel.await_join()
        aggregator_channel.await_join()

        # set necessary properties to help channel decide how to select ends
        trainer_channel.set_property("round", self._round)
        aggregator_channel.set_property("round", self._round)

        logger.info("both mid aggs and trainers joined")

    def pair_mid_aggs_and_trainers(self):
        trainer_channel = self.channel_manager.get_by_tag(TAG_SELECT_AGGREGATOR)
        aggregator_channel = self.channel_manager.get_by_tag(TAG_SELECT_TRAINERS)

        trainer_ends = trainer_channel.ends()
        agg_ends = aggregator_channel.ends()

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

    def send_selected_trainers(self):
        aggregator_channel = self.channel_manager.get_by_tag(TAG_SELECT_TRAINERS)
        if not aggregator_channel:
            return

        for agg, trainers in self.agg_to_trainer.items():
            msg = {MessageType.COORDINATED_ENDS: trainers}
            aggregator_channel.send(agg, msg)

    def send_selected_aggregator(self):
        trainer_channel = self.channel_manager.get_by_tag(TAG_SELECT_AGGREGATOR)

        for trainer, agg in self.trainer_to_agg.items():
            msg = {MessageType.COORDINATED_ENDS: agg}
            trainer_channel.send(trainer, msg)

    def graceful_exit(self):
        logger.info("help mid aggs and trainers to receive EOT")
        self.await_mid_aggs_and_trainers()
        self.pair_mid_aggs_and_trainers()
        self.send_selected_trainers()
        self.send_selected_aggregator()

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

            task_check_eot = Tasklet(self.check_eot)

            task_await = Tasklet(self.await_mid_aggs_and_trainers)

            task_pairing = Tasklet(self.pair_mid_aggs_and_trainers)

            task_send_trainers_to_agg = Tasklet(self.send_selected_trainers)

            task_send_agg_to_trainer = Tasklet(self.send_selected_aggregator)

            task_graceful_exit = Tasklet(self.graceful_exit)

        loop = Loop(loop_check_fn=lambda: self._work_done)
        (
            loop(
                task_check_eot
                >> task_await
                >> task_pairing
                >> task_send_trainers_to_agg
                >> task_send_agg_to_trainer
            )
            >> task_graceful_exit
        )

    def run(self) -> None:
        """Run role."""
        self.composer.run()

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the top level aggregator role."""
        return [TAG_SELECT_TRAINERS, TAG_SELECT_AGGREGATOR, TAG_CHECK_EOT]
