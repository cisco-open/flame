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

import asyncio
import logging

from flame.channel import KEY_CH_SELECT_REQUESTER, KEY_CH_STATE, VAL_CH_STATE_RECV
from flame.channel_manager import ChannelManager
from flame.common.util import background_thread_loop, run_async
from flame.config import Config, SelectorType
from flame.end import KEY_END_STATE, End
from flame.mode.composer import Composer
from flame.mode.horizontal.coord_asyncfl import (
    CHANNEL_MID_TO_TOP_AGG,
    CHANNEL_MID_TO_TRAINER,
    CHANNEL_TOP_TO_MID_AGG,
    CHANNEL_TRAINER_TO_MID,
)
from flame.mode.message import MessageType
from flame.mode.role import Role
from flame.mode.tasklet import Loop, Tasklet
from flame.selectors import selector_provider

logger = logging.getLogger(__name__)

TAG_COORDINATE_WITH_TOP_AGG = "coordinateWithTopAgg"
TAG_COORDINATE_WITH_MID_AGG = "coordinateWithMidAgg"
TAG_COORDINATE_WITH_TRAINER = "coordinateWithTrainer"

TASK_TOP_AGG = "top"
TASK_MID_AGG = "mid"
TASK_TRAINER = "trainer"


class Coordinator(Role):
    """Coordinator for asynchronous FL."""

    def __init__(self, config: Config, **kwargs):
        """Initialize coordinator instance."""
        with background_thread_loop() as loop:
            self._loop = loop

        self.config = config

        self.channel_manager = self.build_channel_manager()

        self._work_done = False

        concurrency = self.config.hyperparameters.concurrency
        logger.debug(f"concurrency: {concurrency}")

        self.top_agg_selector = selector_provider.get(
            SelectorType.FEDBUFF, c=concurrency
        )
        self.mid_agg_selector = selector_provider.get(
            SelectorType.FEDBUFF, c=concurrency
        )

        logger.debug(f"{self.top_agg_selector} {self.mid_agg_selector}")

        self.mid_agg_ends = dict()
        self.trainer_ends = dict()

        self.mid_agg_to_top_agg = dict()
        self.trainer_to_mid_agg = dict()

        # for monitoring purpose of active ends
        self.prev_top_aggs = set()
        self.prev_mid_aggs = set()

        self.tasks = {TASK_TOP_AGG: None, TASK_MID_AGG: None, TASK_TRAINER: None}

        self.task_coros = {
            TASK_TOP_AGG: self._handle_top_agg_req,
            TASK_MID_AGG: self._handle_mid_agg_req,
            TASK_TRAINER: self._handle_trainer_req,
        }

    def build_channel_manager(self):
        """Build and initialize a channel manager."""
        channel_manager = ChannelManager()
        channel_manager(self.config)
        channel_manager.join_all()

        return channel_manager

    async def get_channel(self, tag: str, *, no_wait: bool = False):
        """Return channel of a given tag when it is ready to use."""
        channel = self.channel_manager.get_by_tag(tag)
        if not channel:
            raise ValueError(f"channel not found for tag {tag}")

        if not no_wait:
            await asyncio.to_thread(channel.await_join)

        return channel

    def _is_valid_req(self, msg: dict) -> bool:
        if MessageType.REQ_COORDINATED_ENDS not in msg:
            logger.debug("REQ_COORDINATED_ENDS not found in the messsage")
            return False

        if len(msg[MessageType.REQ_COORDINATED_ENDS]) != 3:
            logger.debug("malformed REQ_COORDINATED_ENDS")
            return False

        return True

    def _update_end_state(
        self,
        data_plane_ends: dict[str, End],
        ctrl_plane_ends: dict[str, End],
        end_states: dict[str, str],
    ) -> None:
        # VAL_END_STATE_RECV is always set in each end in the control plane
        # channel when recv or recv_fifo is called for the end
        #
        # This leads to misbehaior in fedbuff's select logic by considering
        # that VAL_END_STATE_RECV property is set for the data plane
        #
        # To avoid the issue, we maintain a separate dictionary called
        # data_plane_ends

        # reset
        data_plane_ends.clear()
        # populate the whole dictionary
        # This can be expensive
        # TODO: revisit this later
        for end_id in ctrl_plane_ends.keys():
            data_plane_ends[end_id] = End(end_id)

        for end_id, state in end_states.items():
            if end_id in data_plane_ends:
                data_plane_ends[end_id].set_property(KEY_END_STATE, state)

    async def _handle_top_agg_req(self):
        logger.debug("calling _handle_top_agg_req")

        top_agg_ch = await self.get_channel(TAG_COORDINATE_WITH_TOP_AGG)
        mid_agg_ch = await self.get_channel(TAG_COORDINATE_WITH_MID_AGG)

        # assumption: there is only one top aggregator
        requester = top_agg_ch.one_end()
        coro = asyncio.to_thread(top_agg_ch.recv, requester)
        msg, _ = await coro

        if not msg:
            logger.debug("no message")
            self.tasks[TASK_TOP_AGG] = None
            return

        if MessageType.EOT in msg:
            self._work_done = msg[MessageType.EOT]
            logger.debug(f"work_done = {self._work_done}")

        if not self._is_valid_req(msg):
            logger.debug("invalid request")
            self.tasks[TASK_TOP_AGG] = None
            return

        channel_in_question = msg[MessageType.REQ_COORDINATED_ENDS][0]
        state = msg[MessageType.REQ_COORDINATED_ENDS][1]
        end_states = msg[MessageType.REQ_COORDINATED_ENDS][2]

        logger.debug(
            f"ch: {channel_in_question}, state: {state}, end_states: {end_states}"
        )
        if channel_in_question != CHANNEL_TOP_TO_MID_AGG or end_states is None:
            logger.debug(f"ch_in_q: {channel_in_question}; end_states: {end_states}")
            # return an empty response message in case of invalid channel
            res_msg = {MessageType.RES_COORDINATED_ENDS: []}
            top_agg_ch.send(requester, res_msg)
            self.tasks[TASK_TOP_AGG] = None
            return

        logger.debug(f"mid_agg_ch's ends: {mid_agg_ch._ends}")
        # TODO: accessing mid_agg_ch._ends directly is not ideal.
        #       revisit this later
        self._update_end_state(self.mid_agg_ends, mid_agg_ch._ends, end_states)
        ch_props = {KEY_CH_STATE: state, KEY_CH_SELECT_REQUESTER: requester}
        selected_ends = self.top_agg_selector.select(self.mid_agg_ends, ch_props)
        logger.debug(f"selected_ends: {selected_ends}")

        # fedbuff selector returns the entire list of selected ends
        # in its internal state in case of state is VAL_CH_STATE_RECV
        # in this case we need to reset the trainer_to_mid_agg data
        # structure and repopulates it so that the up-to-date list is
        # recorded in that data structure
        #
        # in case that stae is VAL_CH_STATE_SEND, fedbuff only returns
        # a new list of ends that can be used to send a message
        # hence, there is no need to reset the trainer_to_mid_agg
        # data structure
        if state == VAL_CH_STATE_RECV:
            # here we can simply reset the entire dictionary since there is
            # only one requester which is a top aggregatgor
            #
            # this case is different from middle aggregator's case
            # where there can be more than one middle aggregator
            # hence in case of middle aggregator, we need to selectively
            # reset the dictionary by comparing requester with the value
            # in the dictionary
            self.mid_agg_to_top_agg = dict()

        selected_list = list(selected_ends.keys())
        for selected in selected_list:
            self.mid_agg_to_top_agg[selected] = requester

        res_msg = {MessageType.RES_COORDINATED_ENDS: selected_list}
        top_agg_ch.send(requester, res_msg)

        self.tasks[TASK_TOP_AGG] = None
        logger.debug(f"selected_list: {selected_list}")

    async def _handle_mid_agg_req(self):
        logger.debug("calling _handle_mid_agg_req")

        mid_agg_ch = await self.get_channel(TAG_COORDINATE_WITH_MID_AGG)
        trainer_ch = await self.get_channel(TAG_COORDINATE_WITH_TRAINER)

        def blocking_inner():
            msg, metadata = next(mid_agg_ch.recv_fifo(mid_agg_ch.all_ends(), 1))
            return msg, metadata

        msg, metadata = await asyncio.to_thread(blocking_inner)
        requester, _ = metadata

        if not msg:
            logger.debug("no message")
            self.tasks[TASK_MID_AGG] = None
            return

        if not self._is_valid_req(msg):
            logger.debug("invalid request")
            self.tasks[TASK_MID_AGG] = None
            return

        channel_in_question = msg[MessageType.REQ_COORDINATED_ENDS][0]
        state = msg[MessageType.REQ_COORDINATED_ENDS][1]
        end_states = msg[MessageType.REQ_COORDINATED_ENDS][2]

        logger.debug(
            f"ch: {channel_in_question}, state: {state}, end_states: {end_states}"
        )

        res_msg = {MessageType.EOT: self._work_done}
        valid_channels = [CHANNEL_MID_TO_TOP_AGG, CHANNEL_MID_TO_TRAINER]
        if channel_in_question not in valid_channels:
            logger.debug(f"channel {channel_in_question} is not valid")
            # return an empty response message in case of invalid channel
            res_msg[MessageType.RES_COORDINATED_ENDS] = []
            mid_agg_ch.send(requester, res_msg)
            self.tasks[TASK_MID_AGG] = None
            return

        if channel_in_question == CHANNEL_MID_TO_TRAINER:
            logger.debug(f"{requester} requesting mid_agg_to_trainers end")
            if end_states is None:
                logger.debug("end_states is None")
                res_msg[MessageType.RES_COORDINATED_ENDS] = []
                mid_agg_ch.send(requester, res_msg)
                self.tasks[TASK_MID_AGG] = None
                return

            # TODO: accessing trainer_ch._ends directly is not ideal.
            #       revisit this later
            self._update_end_state(self.trainer_ends, trainer_ch._ends, end_states)
            ch_props = {KEY_CH_STATE: state, KEY_CH_SELECT_REQUESTER: requester}
            selected_ends = self.mid_agg_selector.select(self.trainer_ends, ch_props)

            selected_list = list(selected_ends.keys())

            # fedbuff selector returns the entire list of selected ends
            # in its internal state in case of state is VAL_CH_STATE_RECV
            # in this case we need to reset the trainer_to_mid_agg data
            # structure and repopulates it so that the up-to-date list is
            # recorded in that data structure
            #
            # in case that stae is VAL_CH_STATE_SEND, fedbuff only returns
            # a new list of ends that can be used to send a message
            # hence, there is no need to reset the trainer_to_mid_agg
            # data structure
            if state == VAL_CH_STATE_RECV:
                # reset
                for trainer, mid_agg in list(self.trainer_to_mid_agg.items()):
                    if requester != mid_agg:
                        continue

                    del self.trainer_to_mid_agg[trainer]

            for selected in selected_list:
                self.trainer_to_mid_agg[selected] = requester

        else:
            logger.debug(f"{requester} requesting mid_agg_to_top_agg end")
            selected_list = []
            if requester in self.mid_agg_to_top_agg:
                selected_list = [self.mid_agg_to_top_agg[requester]]

        res_msg[MessageType.RES_COORDINATED_ENDS] = selected_list
        mid_agg_ch.send(requester, res_msg)

        self.tasks[TASK_MID_AGG] = None

        logger.debug(f"requester: {requester}, selected_list: {selected_list}")

    async def _handle_trainer_req(self):
        logger.debug("calling _handle_trainer_req")

        trainer_ch = await self.get_channel(TAG_COORDINATE_WITH_TRAINER)

        def blocking_inner():
            msg, metadata = next(trainer_ch.recv_fifo(trainer_ch.all_ends(), 1))
            return msg, metadata

        msg, metadata = await asyncio.to_thread(blocking_inner)
        requester, _ = metadata
        if not msg:
            logger.debug("no message")
            self.tasks[TASK_TRAINER] = None
            return

        if not self._is_valid_req(msg):
            logger.debug("invalid request")
            self.tasks[TASK_TRAINER] = None
            return

        channel_in_question = msg[MessageType.REQ_COORDINATED_ENDS][0]
        res_msg = {MessageType.EOT: self._work_done}
        if channel_in_question != CHANNEL_TRAINER_TO_MID:
            logger.debug(f"wrong channel in quest: {channel_in_question}")
            # return an empty response message in case of invalid channel
            res_msg[MessageType.RES_COORDINATED_ENDS] = []
            trainer_ch.send(requester, res_msg)
            self.tasks[TASK_TRAINER] = None
            return

        selected_list = []
        if requester in self.trainer_to_mid_agg:
            selected_list = [self.trainer_to_mid_agg[requester]]

        res_msg[MessageType.RES_COORDINATED_ENDS] = selected_list
        trainer_ch.send(requester, res_msg)

        self.tasks[TASK_TRAINER] = None
        logger.debug(f"{requester}: selected_list: {selected_list}")

    def coordinate(self):
        """Coordinate requests in an asynchronous fashion."""

        async def _inner():
            for key, task in self.tasks.items():
                if task:
                    continue

                logger.debug(f"creating coordinate task for {key}")
                self.tasks[key] = asyncio.create_task(self.task_coros[key]())

            done, pending = await asyncio.wait(
                list(self.tasks.values()), return_when=asyncio.FIRST_COMPLETED
            )

            logger.debug(f"done = {len(done)}, pending = {len(pending)}")
            # logger.debug(f"tasks: {self.tasks}")

        _, _ = run_async(_inner(), self._loop)
        logger.debug("done coordination with requests")

    def inform_end_of_training(self) -> None:
        """Inform the end of training."""
        # no need to inform EOT to top aggregator since top aggregator
        # informed EOT in the first place
        logger.debug("calling inform_end_of_training")

        async def _inner():
            mid_agg_channel = await self.get_channel(
                TAG_COORDINATE_WITH_MID_AGG, no_wait=True
            )
            mid_agg_channel.broadcast({MessageType.EOT: self._work_done})

            trainer_channel = await self.get_channel(
                TAG_COORDINATE_WITH_TRAINER, no_wait=True
            )
            trainer_channel.broadcast({MessageType.EOT: self._work_done})

        _, _ = run_async(_inner(), self._loop)
        logger.debug("done broadcasting end-of-training")

    def monitor_agg_ends(self):
        """Monitor aggregator ends.

        In case of end's departure, fedbuff's selected ends for the
        end are reset so that those selected ends can be served by the
        other end. For instance, if a middle aggregator crashes, then
        all the trainers associated with that aggregator should be
        remapped to a different middle aggregator. This method carries
        out the task.
        """

        async def _inner():
            # handle top agg departure
            top_agg_ch = await self.get_channel(
                TAG_COORDINATE_WITH_TOP_AGG, no_wait=True
            )
            curr_top_aggs = set(top_agg_ch.all_ends())

            for end_id in self.prev_top_aggs.difference(curr_top_aggs):
                self.top_agg_selector.reset_selected_ends(end_id)

            self.prev_top_aggs = curr_top_aggs

            # handle mid agg departure
            mid_agg_ch = await self.get_channel(
                TAG_COORDINATE_WITH_MID_AGG, no_wait=True
            )
            curr_mid_aggs = set(mid_agg_ch.all_ends())

            for end_id in self.prev_mid_aggs.difference(curr_mid_aggs):
                self.mid_agg_selector.reset_selected_ends(end_id)

            self.prev_mid_aggs = curr_mid_aggs

        _, _ = run_async(_inner(), self._loop)

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

            tl_monitor = Tasklet("monitor", self.monitor_agg_ends)

            tl_coordinate = Tasklet("coordinate", self.coordinate)

            tl_inform_eot = Tasklet("inform_eot", self.inform_end_of_training)

        loop = Loop(loop_check_fn=lambda: self._work_done)
        loop(tl_monitor >> tl_coordinate) >> tl_inform_eot

    def run(self) -> None:
        """Run role."""
        self.composer.run()

        async def _inner():
            for task in asyncio.all_tasks():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.debug(f"successfully cancelled {task.get_name()}")

            logger.debug("done with cleaning up asyncio tasks")

        _ = run_async(_inner(), self._loop)
        self._loop.stop()

        logger.debug("done")

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the top level aggregator role."""
        return [
            TAG_COORDINATE_WITH_TOP_AGG,
            TAG_COORDINATE_WITH_MID_AGG,
            TAG_COORDINATE_WITH_TRAINER,
        ]
