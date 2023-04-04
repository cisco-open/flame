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
"""Oort horizontal FL top level aggregator."""

import logging
import math
from typing import Callable

import torch
from flame.channel import VAL_CH_STATE_SEND
from flame.common.constants import DeviceType
from flame.common.util import weights_to_device
from flame.mode.composer import Composer
from flame.mode.horizontal.syncfl.trainer import TAG_FETCH, TAG_UPLOAD
from flame.mode.horizontal.syncfl.trainer import Trainer as BaseTrainer
from flame.mode.message import MessageType
from flame.mode.tasklet import Loop, Tasklet

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    """Oort Trainer implements an ML training role."""

    def _send_weights(self, tag: str) -> None:
        """
        Send local model weights to the aggregator, and the statistical
        utility information of a trainer for Oort algorithm.

        This method is overriden from one in horizontal trainer
        (..trainer).
        """
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

        delta_weights = self._delta_weights_fn(self.weights, self.prev_weights)

        msg = {
            MessageType.WEIGHTS: weights_to_device(delta_weights, DeviceType.CPU),
            MessageType.DATASET_SIZE: self.dataset_size,
            MessageType.MODEL_VERSION: self._round,
            MessageType.STAT_UTILITY: self._stat_utility,
        }
        channel.send(end, msg)
        logger.debug("sending weights done")

    def init_oort_variables(self) -> None:
        """Initialize Oort variables."""
        self._stat_utility = 0

    def oort_loss(
        self,
        loss_fn: Callable[..., torch.Tensor],
        output: torch.Tensor,
        target: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        """
        Measure the loss of a trainer during training.
        The trainer's statistical utility is measured at epoch 1.
        """
        if epoch == 1:
            loss_list = loss_fn(output, target, reduction="none")
            self._stat_utility += torch.square(loss_list).sum()
            loss = loss_list.mean()
        else:
            loss = loss_fn(output, target)

        return loss

    def normalize_stat_utility(self, epoch) -> None:
        """
        Normalize statistical utility of a trainer based on the size
        of the trainer's datset, at epoch 1.
        """
        if epoch == 1:
            self._stat_utility = len(self.train_loader.dataset) * math.sqrt(
                self._stat_utility / len(self.train_loader.dataset)
            )
        else:
            return

    def reset_stat_utility(self) -> None:
        """Reset the trainer's statistical utility to zero."""
        self._stat_utility = 0

    def compose(self) -> None:
        """Compose role with tasklets."""
        with Composer() as composer:
            self.composer = composer

            task_internal_init = Tasklet("", self.internal_init)

            task_init_oort_variables = Tasklet("", self.init_oort_variables)

            task_load_data = Tasklet("", self.load_data)

            task_init = Tasklet("", self.initialize)

            task_get = Tasklet("", self.get, TAG_FETCH)

            task_train = Tasklet("", self.train)

            task_eval = Tasklet("", self.evaluate)

            task_put = Tasklet("", self.put, TAG_UPLOAD)

            task_save_metrics = Tasklet("", self.save_metrics)

            # create a loop object with loop exit condition function
            loop = Loop(loop_check_fn=lambda: self._work_done)
            (
                task_internal_init
                >> task_init_oort_variables
                >> task_load_data
                >> task_init
                >> loop(
                    task_get >> task_train >> task_eval >> task_put >> task_save_metrics
                )
            )
