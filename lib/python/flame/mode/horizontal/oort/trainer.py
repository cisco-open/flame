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

import inspect
import logging
import math

import torch
from flame.channel import VAL_CH_STATE_SEND
from flame.common.constants import DeviceType
from flame.common.custom_abcmeta import abstract_attribute
from flame.common.util import weights_to_device
from flame.mode.composer import Composer
from flame.mode.horizontal.syncfl.trainer import TAG_FETCH, TAG_UPLOAD
from flame.mode.horizontal.syncfl.trainer import Trainer as BaseTrainer
from flame.mode.message import MessageType
from flame.mode.tasklet import Loop, Tasklet

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    """Oort Trainer implements an ML training role."""

    @abstract_attribute
    def loss_fn(self):
        """Abstract attribute for loss function."""

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

        delta_weights = self.privacy.apply_dp_fn(delta_weights)

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

        if "reduction" not in inspect.signature(self.loss_fn).parameters:
            msg = "Parameter 'reduction' not found in loss function "
            msg += f"'{self.loss_fn.__name__}', which is required for Oort"
            raise TypeError(msg)

    def oort_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        epoch: int,
        batch_idx: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        Measure the loss of a trainer during training.
        The trainer's statistical utility is measured at epoch 1.
        """
        if epoch == 1 and batch_idx == 0:
            if "reduction" in kwargs.keys():
                reduction = kwargs["reduction"]
            else:
                reduction = "mean"  # default reduction policy is mean
            kwargs_wo_reduction = {
                key: value for key, value in kwargs.items() if key != "reduction"
            }

            criterion = self.loss_fn(reduction="none", **kwargs_wo_reduction)
            loss_list = criterion(output, target)
            self._stat_utility += torch.square(loss_list).sum()

            if reduction == "mean":
                loss = loss_list.mean()
            elif reduction == "sum":
                loss = loss_list.sum()
        else:
            criterion = self.loss_fn(**kwargs)
            loss = criterion(output, target)

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

            task_internal_init = Tasklet("internal_init", self.internal_init)

            task_init_oort_variables = Tasklet(
                "init_oort_variables", self.init_oort_variables
            )

            task_load_data = Tasklet("load_data", self.load_data)

            task_init = Tasklet("initialize", self.initialize)

            task_get = Tasklet("fetch", self.get, TAG_FETCH)

            task_train = Tasklet("train", self.train)

            task_eval = Tasklet("evaluate", self.evaluate)

            task_put = Tasklet("upload", self.put, TAG_UPLOAD)

            task_save_metrics = Tasklet("save_metrics", self.save_metrics)

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
