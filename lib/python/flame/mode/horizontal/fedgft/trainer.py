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
"""FedGFT horizontal FL trainer."""

import logging
import time

from flame.channel import VAL_CH_STATE_RECV, VAL_CH_STATE_SEND
from flame.common.util import MLFramework, get_ml_framework_in_use
from flame.mode.composer import Composer
from flame.mode.horizontal.trainer import TAG_FETCH, TAG_UPLOAD
from flame.mode.horizontal.trainer import Trainer as BaseTrainer
from flame.mode.message import MessageType
from flame.mode.tasklet import Loop, Tasklet

logger = logging.getLogger(__name__)

TAG_FETCH_BIAS = "fetchBias"
TAG_UPLOAD_BIAS = "uploadBias"


class Trainer(BaseTrainer):
    """FedGFT Trainer implements an ML training role."""

    def internal_init(self) -> None:
        """Initialize internal state for role."""
        logger.debug("in internal_init")
        ml_framework_in_use = get_ml_framework_in_use()

        # only support pytorch
        if ml_framework_in_use != MLFramework.PYTORCH:
            raise NotImplementedError(
                "supported ml framework not found; "
                f"supported frameworks (for fedgft) are: {[MLFramework.PYTORCH.name.lower()]}"
            )

        super().internal_init()

    def _fetch_bias(self, tag: str) -> None:
        logger.debug("calling _fetch_bias")

        self.fetch_success = False
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found with tag {tag}")
            # we don't want to keep calling this too fast
            # so let's sleep 1 second
            time.sleep(1)
            return

        # this call waits for at least one peer joins this channel
        channel.await_join()

        # one aggregator is sufficient
        end = channel.one_end(VAL_CH_STATE_RECV)
        msg, _ = channel.recv(end)

        if not msg:
            logger.debug("no message received")
            if self._work_done:
                # when the work is done, we cancel continue condition
                # (i.e., we set fetch_success to True)
                self.fetch_success = True
            # we don't want to keep calling this too fast
            # so let's sleep 1 second
            time.sleep(1)
            return

        if MessageType.BIAS in msg:
            # TO DO: implement regularizer side
            self.regularizer.update_bias(msg[MessageType.BIAS])

        self.fetch_success = True
        logger.debug(f"work_done: {self._work_done}, round: {self._round}")

    def _send_bias(self, tag: str) -> None:
        logger.debug("calling _send_bias")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found with {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        # one aggregator is sufficient
        end = channel.one_end(VAL_CH_STATE_SEND)

        msg = {
            MessageType.BIAS: self.regularizer.bias,
            MessageType.DATASET_SIZE: self.dataset_size,
        }
        channel.send(end, msg)
        logger.debug("sending bias done")

    def pre_process(self) -> None:
        """Complete this method before training to update bias parameters (can be user-defined)."""
        self.regularizer.update_local_bias_params(self.model, self.train_loader)
        self.update_metrics({"local_bias": abs(self.regularizer.get_local_bias())})

    def compose(self) -> None:
        """Compose role with tasklets."""
        with Composer() as composer:
            self.composer = composer

            task_internal_init = Tasklet("internal_init", self.internal_init)

            task_load_data = Tasklet("load_data", self.load_data)

            task_init = Tasklet("init", self.initialize)

            task_get = Tasklet("fetch", self.get, TAG_FETCH)
            task_get.set_continue_fn(cont_fn=lambda: not self.fetch_success)

            task_pre_process = Tasklet("pre_process", self.pre_process)

            task_put_bias = Tasklet("upload_bias", self._send_bias, TAG_UPLOAD_BIAS)

            task_get_bias = Tasklet("fetch_bias", self._fetch_bias, TAG_FETCH_BIAS)

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
                    task_get
                    >> task_pre_process
                    >> task_put_bias
                    >> task_get_bias
                    >> task_train
                    >> task_eval
                    >> task_put
                    >> task_save_metrics
                )
            )

    def run(self) -> None:
        """Run role."""
        self.composer.run()

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the trainer role."""
        return [TAG_FETCH, TAG_UPLOAD, TAG_FETCH_BIAS, TAG_UPLOAD_BIAS]
