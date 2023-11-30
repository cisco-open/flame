# Copyright 2022 Cisco Systems, Inc. and its affiliates
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
"""hybrid FL trainer."""

import logging

from flame.channel_manager import ChannelManager
from flame.common.constants import DeviceType
from flame.common.util import weights_to_device, weights_to_model_device
from flame.mode.composer import Composer
from flame.mode.distributed.trainer import Trainer as DistTrainer
from flame.mode.hybrid.top_aggregator import GROUP_UNIDENTIFIED
from flame.mode.message import MessageType
from flame.mode.tasklet import Loop, Tasklet

logger = logging.getLogger(__name__)

TAG_FETCH = "fetch"
TAG_UPLOAD = "upload"
TAG_RING_ALLREDUCE = "ring_allreduce"


class Trainer(DistTrainer):
    """Trainer inherits distributed trainer."""

    def init_cm(self) -> None:
        """Initialize channel manager."""
        self.cm = ChannelManager()
        self.cm(self.config)

        self.ring_channel_name = ""
        for ch_name, ch_config in self.config.channels.items():
            skip = False
            for _, fn_tag_list in ch_config.func_tags.items():
                if TAG_RING_ALLREDUCE in fn_tag_list:
                    skip = True

            # if skip is true, we found a channel that is for ring allreduce
            # trainer needs to join that channel on a on-demand fashion.
            # so, here we skip join.
            if skip:
                self.ring_channel_name = ch_name
                continue

            self.cm.join(ch_name)

    def _member_check(self, tag: str) -> None:
        """
        In our HybridFL, ends in the same group communicate to figure out
        the total_count. Committer end initiates by sending a message in a ring fashion,
        where ends receiving the message adds its dataset size to it. After one cycle,
        the committer end, who knows the total count, again sends the message in a
        ring fashion, to notify all the ends the total count value.
        """
        if tag != TAG_RING_ALLREDUCE:
            return

        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found with tag {tag}")
            return

        if self.can_ring_allreduce():
            if self.is_committer:
                channel.send(self.sendto_id, {MessageType.DATASET_SIZE: 0})

            msg, _ = channel.recv(self.recvfrom_id)
            aggr_size = msg[MessageType.DATASET_SIZE] + self.dataset_size

            channel.send(self.sendto_id, {MessageType.DATASET_SIZE: aggr_size})

            msg, _ = channel.recv(self.recvfrom_id)
            aggr_size = msg[MessageType.DATASET_SIZE]

            if not self.is_committer:
                channel.send(self.sendto_id, {MessageType.DATASET_SIZE: aggr_size})

            self.total_count = aggr_size

        return

    def can_ring_allreduce(self) -> bool:
        """Return true if a ring is formed for ring-allreduce."""
        return self.group_identified and self.group_size > 1

    def get(self, tag: str) -> None:
        """Get data from remote role(s)."""
        if tag == TAG_FETCH:
            self._fetch_weights(tag)

    def _fetch_weights(self, tag: str) -> None:
        logger.debug("calling _fetch_weights from aggregator")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found with tag {tag}")
            return

        # this call waits for at least one peer joins this channel
        channel.await_join()

        # one aggregator is sufficient
        end = channel.one_end()
        msg, _ = channel.recv(end)

        if MessageType.WEIGHTS in msg:
            self.weights = weights_to_model_device(msg[MessageType.WEIGHTS], self.model)
            self._update_model()

        if MessageType.EOT in msg:
            self._work_done = msg[MessageType.EOT]

        if MessageType.ROUND in msg:
            self._round = msg[MessageType.ROUND]

        if MessageType.HYBRID_METADATA in msg:
            hybrid_metadata = msg[MessageType.HYBRID_METADATA]
            if (
                hybrid_metadata["group_id"]
                == self.config.group_association["param-channel"]
            ):
                self.sendto_id = hybrid_metadata["sendto_id"]
                self.recvfrom_id = hybrid_metadata["recvfrom_id"]
                self.rank = hybrid_metadata["rank"]
                self.group_size = hybrid_metadata["size"]
                self.group_identified = True
            elif hybrid_metadata["group_id"] == GROUP_UNIDENTIFIED:
                self.group_identified = False
            else:
                raise ValueError(
                    "group id should be either given or GROUP_UNIDENTIFIED."
                )

        if MessageType.IS_COMMITTER in msg:
            self.is_committer = msg[MessageType.IS_COMMITTER]

        # once global weights were received, let's join ring channel
        # calling join more than once is okay because channel manager
        # checks if a role already joined that channel or not.
        self.cm.join(self.ring_channel_name)

    def put(self, tag: str) -> None:
        """Set data to remote role(s)."""
        if tag == TAG_UPLOAD:
            self._upload_weights(tag)

    def _upload_weights(self, tag: str) -> None:
        logger.debug("calling _upload_weights to aggregator")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found with {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        # one aggregator is sufficient
        end = channel.one_end()

        if not self.can_ring_allreduce():
            # ring was not formed; so, all-reduce didn't take a place.
            # in such a case, each worker should follow the conventional
            # fl procedure by sharing  their local model weights with
            # an aggregator.
            logger.debug("ring was not formed; each shares its weights")
            self._update_weights()
            weights, size = self.weights, self.dataset_size

        # reaching here means that all-reduce took place. So, a committer
        # only needs to share model weights (built from all-reduce) and
        # the total number of data samples used in a ring.
        #
        # non-committers send a dummy message so that the aggregator won't
        # be blocked.
        # TODO: figure out a way not to send a dummy message
        elif self.is_committer:
            logger.debug("sending real weights")
            weights, size = self.weights, self.total_count

        else:
            logger.debug("sending dummy weights")
            weights, size = None, 0

        delta_weights = self._delta_weights_fn(weights, self.prev_weights)

        msg = {
            MessageType.WEIGHTS: weights_to_device(delta_weights, DeviceType.CPU),
            MessageType.DATASET_SIZE: size,
            MessageType.MODEL_VERSION: self._round,
            MessageType.HYBRID_METADATA: {
                "group_id": self.config.group_association["param-channel"]
            },
        }
        channel.send(end, msg)

    def compose(self) -> None:
        """Compose role with tasklets."""
        with Composer() as composer:
            self.composer = composer

            task_init_cm = Tasklet("init_cm", self.init_cm)

            task_internal_init = Tasklet("internal_init", self.internal_init)

            task_load_data = Tasklet("load_data", self.load_data)

            task_init = Tasklet("initialize", self.initialize)

            task_get = Tasklet("fetch", self.get, TAG_FETCH)

            task_member_check = Tasklet(
                "member_check", self._member_check, TAG_RING_ALLREDUCE
            )

            task_allreduce = Tasklet(
                "ring_allreduce", self._ring_allreduce, TAG_RING_ALLREDUCE
            )

            task_train = Tasklet("train", self.train)

            task_eval = Tasklet("evaluate", self.evaluate)

            task_put = Tasklet("upload", self.put, TAG_UPLOAD)

            task_save_metrics = Tasklet("save_metrics", self.save_metrics)

            task_save_params = Tasklet("save_params", self.save_params)

            task_save_model = Tasklet("save_model", self.save_model)

            # create a loop object with loop exit condition function
            loop = Loop(loop_check_fn=lambda: self._work_done)
            (
                task_init_cm
                >> task_internal_init
                >> task_load_data
                >> task_init
                >> loop(
                    task_get
                    >> task_train
                    >> task_member_check
                    >> task_allreduce
                    >> task_eval
                    >> task_save_metrics
                    >> task_put
                )
                >> task_save_params
                >> task_save_model
            )

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the trainer role."""
        return [TAG_FETCH, TAG_UPLOAD, TAG_RING_ALLREDUCE]
