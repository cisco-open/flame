# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License"); you
# may not use this file except in compliance with the License. You may
# obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""Google Speech horizontal FL trainer for PyTorch."""

import ast
import calendar
import gc
import logging
import os
import threading
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import torchvision.models as models
from flame.config import Config
from flame.mode.horizontal.trainer import Trainer
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS

LABEL_MAP = {
    "backward": 0,
    "bed": 1,
    "bird": 2,
    "cat": 3,
    "dog": 4,
    "down": 5,
    "eight": 6,
    "five": 7,
    "follow": 8,
    "forward": 9,
    "four": 10,
    "go": 11,
    "happy": 12,
    "house": 13,
    "learn": 14,
    "left": 15,
    "marvin": 16,
    "nine": 17,
    "no": 18,
    "off": 19,
    "on": 20,
    "one": 21,
    "right": 22,
    "seven": 23,
    "sheila": 24,
    "six": 25,
    "stop": 26,
    "three": 27,
    "tree": 28,
    "two": 29,
    "up": 30,
    "visual": 31,
    "wow": 32,
    "yes": 33,
    "zero": 34,
}


logger = logging.getLogger(__name__)


class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet34_1D(nn.Module):
    def __init__(self, n_input=1, n_output=35):
        super(ResNet34_1D, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv1d(
            n_input, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, n_output)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )

        layers = []
        layers.append(BasicBlock1D(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class PyTorchSpeechCommandsTrainer(Trainer):
    """PyTorch Speech Commands Trainer."""

    def __init__(self, config: Config) -> None:
        """Initialize a class instance."""
        self.config = config
        self.dataset_size = 0
        self.model = None
        self.model_arch = ResNet34_1D
        # Oort requires its loss function to have 'reduction'
        # parameter
        self.loss_fn = torch.nn.CrossEntropyLoss

        self.device = None
        self.train_loader = None

        self.learning_rate = self.config.hyperparameters.learning_rate
        self.epochs = self.config.hyperparameters.epochs
        self.batch_size = self.config.hyperparameters.batch_size or 16
        self.trainer_id = self.config.task_id

        self.criterion = None

        # Enable/disable use of oort_loss fromt he config. Needed for
        # oort and asyncOORT.
        self.use_oort_loss_fn = self.config.hyperparameters.use_oort_loss_fn
        logger.info(
            f"Trainer: {self.trainer_id} has "
            f"use_oort_loss_fn: {self.use_oort_loss_fn}"
        )

        # TODO: (DG) Remove the hard requirement for config to include
        # trainer_indices_list and failure_durations_s Setting the
        # indices used by the trainer
        self.trainer_indices_list = self.config.hyperparameters.trainer_indices_list
        # Loading the failure durations for trainers
        self.trainer_start_ts = time.time()
        self.failure_durations_s = ast.literal_eval(
            self.config.hyperparameters.failure_durations_s
        )
        if len(self.failure_durations_s) > 0:
            self.timestamp_next_sleep_s = (
                self.trainer_start_ts + self.failure_durations_s[0][0]
            )
            print(
                f"# Trainer id: {self.trainer_id}, self.failure_durations_s: "
                f"{self.failure_durations_s}"
            )
        else:
            self.timestamp_next_sleep_s = calendar.timegm(
                time.strptime("Dec 31, 2030 @ 23:59:59 UTC", "%b %d, %Y @ %H:%M:%S UTC")
            )

        # TODO: (DG) Fix the hack later. Creating duplicate data
        # struct for dup_check_and_sleep() which is used by heartbeat
        # thread
        self.dup_failure_durations_s = ast.literal_eval(
            self.config.hyperparameters.failure_durations_s
        )
        if len(self.dup_failure_durations_s) > 0:
            self.dup_timestamp_next_sleep_s = (
                self.trainer_start_ts + self.dup_failure_durations_s[0][0]
            )
            print(
                f"# Trainer id: {self.trainer_id}, self.dup_failure_durations_s: "
                f"{self.dup_failure_durations_s}"
            )
        else:
            self.dup_timestamp_next_sleep_s = calendar.timegm(
                time.strptime("Dec 31, 2030 @ 23:59:59 UTC", "%b %d, %Y @ %H:%M:%S UTC")
            )

        # sending heartbeats to aggregator
        if "enabled" in self.config.hyperparameters.heartbeats.keys():
            self.heartbeats_enabled = self.config.hyperparameters.heartbeats["enabled"]
        else:
            self.heartbeats_enabled = False

        if "frequency_s" in self.config.hyperparameters.heartbeats.keys():
            self.heartbeats_second_freq = self.config.hyperparameters.heartbeats[
                "frequency_s"
            ]
        else:
            self.heartbeats_second_freq = 99999

        # TODO: (DG) self.timestamp_next_heartbeat_s might not be
        # getting used. Remove? if heartbeats are enabled, compute
        # first heartbeat time
        if self.heartbeats_enabled is True:
            self.timestamp_next_heartbeat_s = (
                self.trainer_start_ts + self.heartbeats_second_freq
            )
        else:
            self.timestamp_next_heartbeat_s = calendar.timegm(
                time.strptime("Dec 31, 2030 @ 23:59:59 UTC", "%b %d, %Y @ %H:%M:%S UTC")
            )

        # Check if client will notify aggregator of its availability
        self.client_avail_aware_notify = (
            self.config.hyperparameters.client_avail_aware_notify
        )

        # Check if client will emulate delays in training time
        self.training_delay_enabled = self.config.hyperparameters.training_delay_enabled
        self.training_delay_s = float(self.config.hyperparameters.training_delay_s)

    def check_and_sleep(self):
        """Induce transient unavailability"""
        # Implement this if transient unavailability need to be
        # emulated in the trainer

        if (time.time() >= self.timestamp_next_sleep_s) and (
            len(self.failure_durations_s) > 0
        ):
            # pop leftmost element
            sleep_config_tuple = self.failure_durations_s.pop(0)

            # get the duration of sleep and set the params for next
            # sleep
            sleep_start_ts_from_trainer_init = (
                self.trainer_start_ts + sleep_config_tuple[0]
            )
            sleep_duration_s = sleep_config_tuple[1]

            # remaining sleep = trainer_start_ts + actual_sleep_start
            # + actual_sleep_duration - current_ts
            remaining_sleep_duration_s = (
                sleep_start_ts_from_trainer_init + sleep_duration_s - time.time()
            )
            logger.debug(
                f"Task_id: {self.trainer_id} given_sleep_duration_s: "
                f"{sleep_duration_s} with remaining_sleep_duration_s: "
                f"{remaining_sleep_duration_s} at timestamp: {time.time()}"
            )

            if remaining_sleep_duration_s <= 0:
                logger.info(
                    f"Task_id: {self.trainer_id} got -ve remaining sleep "
                    f"at timestamp: {time.time()}"
                )
                # Need to pop out failure intervals that occur in the
                # past
                time_elapsed_from_start = time.time() - self.trainer_start_ts
                while len(self.failure_durations_s) > 0 and (
                    time_elapsed_from_start
                    > (self.failure_durations_s[0][0] + self.failure_durations_s[0][1])
                ):
                    self.failure_durations_s.pop(0)
                    if len(self.failure_durations_s) == 0:
                        break
            else:
                # sleep for remaining time
                logger.info(
                    f"Task_id: {self.trainer_id} going to sleep "
                    f"at timestamp: {time.time()}"
                )
                time.sleep(remaining_sleep_duration_s)
                logger.info(
                    f"Task_id: {self.trainer_id} woke up at timestamp: "
                    f"{time.time()}"
                )

            # check if failure_list is now empty, if yes, reset
            # ts_next_sleep_s if not empty, set it to the next value
            if len(self.failure_durations_s) > 0:
                self.timestamp_next_sleep_s = max(
                    (self.trainer_start_ts + self.failure_durations_s[0][0]),
                    time.time() + 1,
                )
                if self.timestamp_next_sleep_s < time.time():
                    logger.error(
                        f"Task_id: {self.trainer_id} ERROR - JUST SET NEXT "
                        f"self.timestamp_next_sleep_s < time.time()"
                    )
            else:
                self.timestamp_next_sleep_s = calendar.timegm(
                    time.strptime(
                        "Dec 31, 2030 @ 23:59:59 UTC", "%b %d, %Y @ %H:%M:%S UTC"
                    )
                )
                logger.info(f"Task_id: {self.trainer_id} no more sleep for trainer")

        logger.debug(
            f"Task_id: {self.trainer_id} check_and_sleep completed at "
            f"timestamp: {time.time()}"
        )

    def dup_check_and_sleep(self):

        if (time.time() >= self.dup_timestamp_next_sleep_s) and (
            len(self.dup_failure_durations_s) > 0
        ):
            # pop leftmost element
            sleep_config_tuple = self.dup_failure_durations_s.pop(0)

            # get the duration of sleep and set the params for next
            # sleep
            sleep_start_ts_from_trainer_init = (
                self.trainer_start_ts + sleep_config_tuple[0]
            )
            sleep_duration_s = sleep_config_tuple[1]

            # remaining sleep = trainer_start_ts + actual_sleep_start
            # + actual_sleep_duration - current_ts
            remaining_sleep_duration_s = (
                sleep_start_ts_from_trainer_init + sleep_duration_s - time.time()
            )
            logger.debug(
                f"Task_id: {self.trainer_id} given_sleep_duration_s: "
                f"{sleep_duration_s} with remaining_sleep_duration_s: "
                f"{remaining_sleep_duration_s} at timestamp: {time.time()}"
            )

            if remaining_sleep_duration_s <= 0:
                logger.info(
                    f"Task_id: {self.trainer_id} got -ve remaining sleep at "
                    f"timestamp: {time.time()}"
                )
                # Need to pop out failure intervals that occur in the
                # past
                time_elapsed_from_start = time.time() - self.trainer_start_ts
                while len(self.dup_failure_durations_s) > 0 and (
                    time_elapsed_from_start
                    > (
                        self.dup_failure_durations_s[0][0]
                        + self.dup_failure_durations_s[0][1]
                    )
                ):
                    self.dup_failure_durations_s.pop(0)
                    if len(self.dup_failure_durations_s) == 0:
                        break
            else:
                # sleep for remaining time
                logger.info(
                    f"Task_id: {self.trainer_id} going to sleep "
                    f"at timestamp: {time.time()}"
                )
                time.sleep(remaining_sleep_duration_s)
                logger.info(
                    f"Task_id: {self.trainer_id} woke up at timestamp: "
                    f"{time.time()}"
                )

            # check if failure_list is now empty, if yes, reset
            # ts_next_sleep_s if not empty, set it to the next value
            if len(self.dup_failure_durations_s) > 0:
                self.dup_timestamp_next_sleep_s = max(
                    (self.trainer_start_ts + self.dup_failure_durations_s[0][0]),
                    time.time() + 1,
                )
                if self.dup_timestamp_next_sleep_s < time.time():
                    logger.error(
                        f"Task_id: {self.trainer_id} ERROR - JUST SET NEXT "
                        f"self.dup_timestamp_next_sleep_s < time.time()"
                    )
            else:
                self.dup_timestamp_next_sleep_s = calendar.timegm(
                    time.strptime(
                        "Dec 31, 2030 @ 23:59:59 UTC", "%b %d, %Y @ %H:%M:%S UTC"
                    )
                )
                logger.info(f"Task_id: {self.trainer_id} no more sleep for trainer")

        logger.debug(
            f"Task_id: {self.trainer_id} dup_check_and_sleep completed at "
            f"timestamp: {time.time()}"
        )

    def check_leave_sleep_join(self):
        """Indicate transient unavailability to aggregator"""
        # NOTE: Builds on top of dup_check_and_sleep, both cannot be
        # used together

        # Condition 1: If current time >= time for next
        # unavailability, leave channel and go to sleep
        if (time.time() >= self.dup_timestamp_next_sleep_s) and (
            len(self.dup_failure_durations_s) > 0
        ):
            # pop leftmost element
            sleep_config_tuple = self.dup_failure_durations_s.pop(0)

            # get the duration of sleep and set the params for next
            # sleep
            sleep_start_ts_from_trainer_init = (
                self.trainer_start_ts + sleep_config_tuple[0]
            )
            sleep_duration_s = sleep_config_tuple[1]

            # remaining sleep = trainer_start_ts + actual_sleep_start
            # + actual_sleep_duration - current_ts
            remaining_sleep_duration_s = (
                sleep_start_ts_from_trainer_init + sleep_duration_s - time.time()
            )
            logger.debug(
                f"Task_id: {self.trainer_id} given_sleep_duration_s: "
                f"{sleep_duration_s} with remaining_sleep_duration_s: "
                f"{remaining_sleep_duration_s}"
                f" at timestamp: {time.time()}"
            )

            if remaining_sleep_duration_s < 0:
                logger.info(
                    f"Task_id: {self.trainer_id} got -ve remaining sleep "
                    f"at timestamp: {time.time()}"
                )
                # Need to pop out failure intervals that occur in the
                # past
                while len(self.dup_failure_durations_s) > 0 and (
                    (time.time() - self.trainer_start_ts)
                    > (
                        self.dup_failure_durations_s[0][0]
                        + self.dup_failure_durations_s[0][1]
                    )
                ):
                    self.dup_failure_durations_s.pop(0)
                    if len(self.dup_failure_durations_s) == 0:
                        break
            else:
                # leave channel, if notify is enabled
                if self.client_avail_aware_notify == "True":
                    self._perform_channel_leave(tag="upload")

                # sleep for remaining time
                logger.info(
                    f"Task_id: {self.trainer_id} going to sleep "
                    f"at timestamp: {time.time()}"
                )
                time.sleep(remaining_sleep_duration_s)
                logger.info(
                    f"Task_id: {self.trainer_id} woke up at timestamp: "
                    f"{time.time()}"
                )

                # join channel, if notify is enabled
                if self.client_avail_aware_notify == "True":
                    self._perform_channel_join(tag="upload")

            # check if failure_list is now empty, if yes, reset
            # ts_next_sleep_s if not empty, set it to the next value
            if len(self.dup_failure_durations_s) > 0:
                self.dup_timestamp_next_sleep_s = max(
                    (self.trainer_start_ts + self.dup_failure_durations_s[0][0]),
                    time.time() + 1,
                )
                if self.dup_timestamp_next_sleep_s < time.time():
                    logger.error(
                        f"Task_id: {self.trainer_id} ERROR - JUST SET NEXT "
                        f"self.dup_timestamp_next_sleep_s < time.time()"
                    )
                else:
                    # We have a next time for unavailability. To
                    # reduce system load, thread can go to sleep until
                    # then and wake up just before the next
                    # unavailability needs to be announced.
                    logger.info(
                        f"Trainer {self.trainer_id} at time {time.time()} set "
                        f"next dup_timestamp_next_sleep_s for "
                        f"{self.dup_timestamp_next_sleep_s}."
                    )
                    self.remaining_time_before_next_unavail_s = (
                        self.dup_timestamp_next_sleep_s - time.time()
                    )
                    logger.debug(
                        f"Trainer {self.trainer_id} will go to sleep for "
                        f"{self.remaining_time_before_next_unavail_s-1} since "
                        f"remaining time before next unavail is "
                        f"{self.remaining_time_before_next_unavail_s}"
                    )
                    thread_rest_duration_s = max(
                        0, (self.remaining_time_before_next_unavail_s - 1)
                    )
                    time.sleep(thread_rest_duration_s)
                    logger.debug(
                        f"Trainer {self.trainer_id} got up from rest at "
                        f"time {time.time()}"
                    )
            else:
                self.dup_timestamp_next_sleep_s = calendar.timegm(
                    time.strptime(
                        "Dec 31, 2030 @ 23:59:59 UTC", "%b %d, %Y @ %H:%M:%S UTC"
                    )
                )
                logger.info(
                    f"Task_id: {self.trainer_id} no more sleep for trainer. "
                    f"Thread will be put to rest for 7 days."
                )
                thread_rest_duration_s = 7 * 24 * 60 * 60
                time.sleep(thread_rest_duration_s)
                logger.debug(
                    f"Trainer {self.trainer_id} got up from rest at "
                    f"time {time.time()}"
                )

        # Condition 2: If current time < time for next
        # unavailability, put the thread to rest
        elif self.dup_timestamp_next_sleep_s > (time.time() + 1):
            self.remaining_time_before_next_unavail_s = (
                self.dup_timestamp_next_sleep_s - time.time()
            )
            thread_rest_duration_s = max(
                0,
                min((self.remaining_time_before_next_unavail_s - 1), 7 * 24 * 60 * 60),
            )
            logger.info(
                f"Outer check_leave_sleep_join check for trainer "
                f"{self.trainer_id}. Next sleep ts is "
                f"{self.dup_timestamp_next_sleep_s}, will "
                f"rest for {thread_rest_duration_s}"
            )
            time.sleep(thread_rest_duration_s)

        logger.debug(
            f"Task_id: {self.trainer_id} check_leave_sleep_join completed at "
            f"timestamp: {time.time()}"
        )

    def initialize(self) -> None:
        """Initialize role."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model_arch().to(self.device)
        logger.debug(
            f"Task_id: {self.trainer_id} initialize completed at timestamp: "
            f"{time.time()}. Skipped model load."
        )

    def load_data(self) -> None:
        """Load data."""
        data_dir = "./data"
        os.makedirs(data_dir, exist_ok=True)

        dataset = SPEECHCOMMANDS(data_dir, download=True, subset="training")

        def custom_collate_fn(batch):
            # Extract sequences and labels
            sequences = [
                item[0].squeeze(0) for item in batch
            ]  # Remove the extra dimension
            labels = [item[2] for item in batch]  # Assuming item[2] is the label

            # Convert string labels to integers using the label map
            labels = [LABEL_MAP[label] for label in labels]

            # Determine the maximum length in this batch
            max_length = max(sequence.size(0) for sequence in sequences)

            # Pad sequences to the maximum length
            padded_sequences = torch.zeros(
                (len(sequences), 1, max_length)
            )  # Add channel dimension
            for i, sequence in enumerate(sequences):
                padded_sequences[i, 0, : sequence.size(0)] = sequence

            # Convert labels to tensor
            labels = torch.tensor(labels)

            return padded_sequences, labels

        # Check if trainer_indices_list is empty
        if self.trainer_indices_list:
            logger.debug(
                f"Value of self.trainer_indices_list: {self.trainer_indices_list}"
            )
            # Create indices into a list and convert to tensor
            indices = torch.tensor(self.trainer_indices_list)
            dataset = data_utils.Subset(dataset, indices)

        train_kwargs = {
            "batch_size": self.batch_size,
            "drop_last": True,
            "shuffle": True,
            "num_workers": 2,
            "collate_fn": custom_collate_fn,
        }

        self.train_loader = DataLoader(dataset, **train_kwargs)

        # Store data into dataset for analysis (e.g., bias)
        # self.dataset = Dataset(dataloader=self.train_loader)

        # Release the memory of the full dataset
        del dataset
        gc.collect()

        logger.info(
            f"Task_id: {self.trainer_id} load_data completed at timestamp: "
            f"{time.time()}, with {len(self.train_loader)} samples"
        )

    def train(self) -> None:
        """Train a model."""
        gpu_train_start_time = time.time()
        # Load model onto GPU
        if self.model is None:
            logger.error("Inside train() but model is None!")
            return

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        # reset stat utility for OORT
        self.reset_stat_utility()

        for epoch in range(1, self.epochs + 1):
            self._train_epoch(epoch)

        # save dataset size so that the info can be shared with
        # aggregator
        self.dataset_size = len(self.train_loader.dataset)

        gpu_train_end_time = time.time()
        actual_gpu_train_time_s = gpu_train_end_time - gpu_train_start_time
        logger.info(
            f"Actual GPU training time for trainer "
            f"{self.trainer_id} is {actual_gpu_train_time_s}s"
        )

        # emulate delays in training (due to compute resource and/or
        # dataset size and/or network latency) if enabled
        if self.training_delay_enabled == "True":
            remaining_time_delay_s = self.training_delay_s - actual_gpu_train_time_s
            if remaining_time_delay_s > 0:
                time.sleep(remaining_time_delay_s)
                logger.debug(
                    f"Delayed training time for trainer "
                    f"{self.trainer_id} by {remaining_time_delay_s} to get "
                    f"total delay of {self.training_delay_s}s"
                )
            else:
                logger.warning(
                    f"GPU training time for "
                    f"{self.trainer_id} was {actual_gpu_train_time_s}. It "
                    f"exceedes the designated delay of "
                    f"{self.training_delay_s}s"
                )

    def _train_epoch(self, epoch):
        if self.model is None:
            logger.error("Inside _train_epoch() but model is None!")
            return

        self.model.train()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)

            if self.use_oort_loss_fn == "False":
                # Loss function to use with Fedbuff
                loss = F.nll_loss(output, target)
            elif self.use_oort_loss_fn == "True":
                # Calculate statistical utility of a trainer while
                # calculating loss
                loss = self.oort_loss(output, target, epoch, batch_idx)

            loss.backward()

            self.optimizer.step()
            if batch_idx % 100 == 0:
                done = batch_idx * len(data)
                total = len(self.train_loader.dataset)
                percent = 100.0 * batch_idx / len(self.train_loader)
                logger.info(
                    f"epoch: {epoch} [{done}/{total} ({percent:.0f}%)]"
                    f"\tloss: {loss.item():.6f}"
                )

        # Explicitly free GPU memory after processing all batches
        # del data, target
        # torch.cuda.empty_cache()
        # gc.collect()  # Force garbage collection
        # torch.cuda.empty_cache()  # Clear the CUDA cache again, just in case

        # normalize statistical utility of a trainer based on the size
        # of the dataset
        self.normalize_stat_utility(epoch)

    def evaluate(self) -> None:
        """Evaluate a model."""
        # Implement this if testing is needed in trainer
        pass

    def initiate_heartbeat(self) -> None:
        while True:
            # heartbeats are sent from a different thread. Ideally
            # heartbeats and sleep should have happened on the same
            # thread but in the current scenario, both threads need to
            # be put to sleep whenever the trainer is marked to be
            # unavailable.

            # issue: if i use check_and_sleep here as well, it will
            # modify existing data struct HACK: duplicate
            # check_and_sleep as dup_check_and_sleep and operate on a
            # duplicate data structure

            # TODO: DG Need to fix that arg isnt being used to
            # enable/disable this
            time.sleep(self.heartbeats_second_freq)
            self.dup_check_and_sleep()
            logger.debug("Initiating send heartbeat to aggregator")
            self.send_heartbeat_to_agg()

    def notify_trainer_avail(self) -> None:
        while True:
            # Adopted from initiate heartbeats

            time.sleep(0.1)  # Will check every 0.1 second
            self.check_leave_sleep_join()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("config", nargs="?", default="./config.json")

    args = parser.parse_args()
    config = Config(args.config)

    t = PyTorchSpeechCommandsTrainer(config)
    print(
        f"# Trainer id: {t.trainer_id}, has heartbeats_enabled: "
        f"{t.heartbeats_enabled}, has client_avail_aware_notify: "
        f"{t.client_avail_aware_notify}, has "
        f"training_delay_enabled: {t.training_delay_enabled}, "
        f"with training_delay_s: {t.training_delay_s}"
    )

    if t.heartbeats_enabled == "True":
        logger.info(
            f"Will initiate thread to send heartbeats for " f"trainer {t.trainer_id}"
        )
        heartbeat_thread = threading.Thread(target=t.initiate_heartbeat)
        heartbeat_thread.daemon = True
        heartbeat_thread.start()
    elif t.client_avail_aware_notify == "True":
        logger.info(
            f"Will initiate thread to send avail notifications for "
            f"trainer {t.trainer_id}"
        )
        avail_notify_thread = threading.Thread(target=t.notify_trainer_avail)
        avail_notify_thread.daemon = True
        avail_notify_thread.start()

    t.compose()
    t.run()


if __name__ == "__main__":
    main()
