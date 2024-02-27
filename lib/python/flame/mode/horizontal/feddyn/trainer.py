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
"""FedDyn horizontal FL trainer."""

import logging

from flame.channel import VAL_CH_STATE_RECV, VAL_CH_STATE_SEND
from flame.common.constants import DeviceType, TrainState
from flame.common.util import (
    MLFramework,
    get_ml_framework_in_use,
    weights_to_device,
    weights_to_model_device,
)
from flame.mode.composer import Composer
from flame.mode.horizontal.trainer import Trainer as BaseTrainer
from flame.mode.message import MessageType
from flame.mode.tasklet import Loop, Tasklet

logger = logging.getLogger(__name__)

TAG_FETCH = "fetch"
TAG_UPLOAD = "upload"
TAG_UPLOAD_DATASET_SIZE = "uploadDatasetSize"


class Trainer(BaseTrainer):
    """FedDyn Trainer implements an ML training role."""

    def internal_init(self) -> None:
        """Initialize internal state for role."""
        logger.debug("in internal_init")
        ml_framework_in_use = get_ml_framework_in_use()

        # only support pytorch
        if ml_framework_in_use != MLFramework.PYTORCH:
            raise NotImplementedError(
                "supported ml framework not found; "
                f"supported frameworks (for feddyn) are: {[MLFramework.PYTORCH.name.lower()]}"
            )

        super().internal_init()

    def get(self, tag: str) -> None:
        """Get data from remote role(s)."""
        if tag == TAG_FETCH:
            self._fetch_weights(tag)

    def _fetch_weights(self, tag: str) -> None:
        logger.debug("calling _fetch_weights")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"[_fetch_weights] channel not found with tag {tag}")
            return

        # this call waits for at least one peer joins this channel
        channel.await_join()

        # one aggregator is sufficient
        end = channel.one_end(VAL_CH_STATE_RECV)
        msg, _ = channel.recv(end)

        if MessageType.WEIGHTS in msg:
            self.weights = weights_to_model_device(msg[MessageType.WEIGHTS], self.model)
            self._update_model()

        # adjust alpha based on aggregator computation
        if MessageType.ALPHA_ADPT in msg:
            logger.debug("got alpha_adpt")
            self.regularizer.alpha = msg[MessageType.ALPHA_ADPT]

        if MessageType.EOT in msg:
            self._work_done = msg[MessageType.EOT]

        if MessageType.ROUND in msg:
            self._round = msg[MessageType.ROUND]

        self.regularizer.save_state(TrainState.PRE, glob_model=self.model)
        logger.debug(f"work_done: {self._work_done}, round: {self._round}")

    def put(self, tag: str) -> None:
        """Set data to remote role(s)."""
        if tag == TAG_UPLOAD:
            self._send_weights(tag)
        elif tag == TAG_UPLOAD_DATASET_SIZE:
            self._send_dataset_size(tag)

    def _send_weights(self, tag: str) -> None:
        logger.debug("calling _send_weights")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"[_send_weights] channel not found with {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        # one aggregator is sufficient
        end = channel.one_end(VAL_CH_STATE_SEND)

        self._update_weights()
        self.regularizer.save_state(TrainState.POST, loc_model=self.model)

        delta_weights = self._delta_weights_fn(self.weights, self.prev_weights)

        delta_weights = self.privacy.apply_dp_fn(delta_weights)

        # send delta_weights to regularizer
        self.regularizer.update()

        msg = {
            MessageType.WEIGHTS: weights_to_device(delta_weights, DeviceType.CPU),
            MessageType.DATASET_SIZE: self.dataset_size,
            MessageType.MODEL_VERSION: self._round,
            MessageType.DATASAMPLER_METADATA: self.datasampler.get_metadata(),
        }
        channel.send(end, msg)
        logger.debug("sending weights done")

    def _send_dataset_size(self, tag: str) -> None:
        logger.debug("calling _send_dataset_size")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"[_send_dataset_size] channel not found with {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        # one aggregator is sufficient
        end = channel.one_end(VAL_CH_STATE_SEND)

        msg = {MessageType.DATASET_SIZE: self.dataset_size}
        logger.debug(f"msg data : {msg}")
        channel.send(end, msg)
        logger.debug("sending dataset size done")

    def compose(self) -> None:
        """Compose role with tasklets."""
        logger.debug("in compose")
        with Composer() as composer:
            self.composer = composer

            task_internal_init = Tasklet("internal_init", self.internal_init)

            task_load_data = Tasklet("load_data", self.load_data)

            task_init = Tasklet("init", self.initialize)

            task_put_dataset_size = Tasklet(
                "upload_dataset_size", self.put, TAG_UPLOAD_DATASET_SIZE
            )

            task_get = Tasklet("fetch", self.get, TAG_FETCH)

            task_train = Tasklet("train", self.train)

            task_eval = Tasklet("evaluate", self.evaluate)

            task_put = Tasklet("upload", self.put, TAG_UPLOAD)

            task_save_metrics = Tasklet("save_metrics", self.save_metrics)

            # create a loop object with loop exit condition function
            loop = Loop(loop_check_fn=lambda: self._work_done)
            (
                task_internal_init
                >> task_load_data
                >> task_init
                >> loop(
                    task_put_dataset_size
                    >> task_get
                    >> task_train
                    >> task_eval
                    >> task_put
                    >> task_save_metrics
                )
            )

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the trainer role."""
        return [TAG_FETCH, TAG_UPLOAD, TAG_UPLOAD_DATASET_SIZE]
