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
"""CIFAR-10 horizontal FL trainer for PyTorch.

The example below is implemented based on the following example from
pytorch:
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html.
"""

import ast
import calendar
import gc
import logging
import os
import threading
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import torchvision.transforms as transforms
from flame.config import Config, TrainerAvailState
from flame.mode.horizontal.trainer import Trainer
from torchvision.datasets import CIFAR10

logger = logging.getLogger(__name__)


class Net(nn.Module):
    """Net class."""

    def __init__(self):
        """Initialize."""
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        """Forward."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class PyTorchCifar10Trainer(Trainer):
    """PyTorch CIFAR-10 Trainer."""

    def __init__(self, config: Config, battery_threshold, speedup_factor) -> None:
        """Initialize a class instance."""
        self.config = config
        self.dataset_size = 0
        self.model = None
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

        self.task_to_perform = "train"

        # Enable/disable use of oort_loss fromt he config. Needed for
        # oort and asyncOORT.
        self.use_oort_loss_fn = self.config.hyperparameters.use_oort_loss_fn
        logger.info(
            f"Trainer: {self.trainer_id} has "
            f"use_oort_loss_fn: {self.use_oort_loss_fn}"
        )

        # TODO: (DG) Remove the hard requirement for config to include
        # trainer_indices_list
        # Setting the indices used by the trainer
        self.trainer_indices_list = self.config.hyperparameters.trainer_indices_list
        self.trainer_start_ts = time.time()

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
        self.client_notify = self.config.hyperparameters.client_notify

        # Check if client will emulate delays in training time
        self.training_delay_enabled = self.config.hyperparameters.training_delay_enabled
        self.training_delay_s = float(self.config.hyperparameters.training_delay_s)

        # Set speedup factor to accelerate all events and training/
        # eval durations
        self.speedup_factor = speedup_factor

        # Use the battery_threshold to determine the
        # avl_events_3_state config. Default to 50 if not provided
        self.event_battery_threshold = battery_threshold
        logger.info(
            f"Trainer id {self.trainer_id} has battery threshold set to {self.event_battery_threshold}"
        )
        
        # Helper function to handle both string and list formats
        def parse_trace(value):
            if isinstance(value, list):
                return value  # Already parsed (from JSON config)
            else:
                # String format (from file config) - validate it's a safe list literal
                try:
                    parsed = ast.literal_eval(value)
                    if not isinstance(parsed, list):
                        raise ValueError(f"Expected list, got {type(parsed)}")
                    return parsed
                except (ValueError, SyntaxError) as e:
                    raise ValueError(f"Invalid trace format: {e}")
        
        if self.event_battery_threshold == 50:
            self.avl_events_3_state = parse_trace(
                self.config.hyperparameters.avl_events_mobiperf_3st_50
            )
        elif self.event_battery_threshold == 75:
            self.avl_events_3_state = parse_trace(
                self.config.hyperparameters.avl_events_mobiperf_3st_75
            )

        self.avl_events_mobiperf_2st = parse_trace(
            self.config.hyperparameters.avl_events_mobiperf_2st
        )

        # Storing synthetic avail traces
        self.avl_events_syn_0 = parse_trace(
            self.config.hyperparameters.avl_events_syn_0
        )

        self.avl_events_syn_20 = parse_trace(
            self.config.hyperparameters.avl_events_syn_20
        )

        self.avl_events_syn_50 = parse_trace(
            self.config.hyperparameters.avl_events_syn_50
        )

        if self.client_notify["trace"] == "mobiperf_3st":
            self.state_avl_event_ts = self.avl_events_3_state
            logger.info(
                f"Set avl_events_3_state for trainer id {self.trainer_id} using battery threshold {self.event_battery_threshold}"
            )
        elif self.client_notify["trace"] == "mobiperf_2st":
            self.state_avl_event_ts = self.avl_events_mobiperf_2st
            logger.info(
                f"Set avl_events_mobiperf_2st for trainer id {self.trainer_id}."
            )
        elif self.client_notify["trace"] == "syn_0":
            self.state_avl_event_ts = self.avl_events_syn_0
            logger.info(f"Set avl_events_syn_0 for trainer id {self.trainer_id}.")
        elif self.client_notify["trace"] == "syn_20":
            self.state_avl_event_ts = self.avl_events_syn_20
            logger.info(f"Set avl_events_syn_20 for trainer id {self.trainer_id}.")
        elif self.client_notify["trace"] == "syn_50":
            self.state_avl_event_ts = self.avl_events_syn_50
            logger.info(f"Set avl_events_syn_50 for trainer id {self.trainer_id}.")
        else:
            logger.info(
                f"No avl_events set for trainer id {self.trainer_id} since state not specified."
            )

        self.avl_state = TrainerAvailState.AVL_TRAIN

        # flag to decide whether the trainer upon unavailability will wait or exit
        self.wait_until_next_avl = self.config.hyperparameters.wait_until_next_avl

    def check_and_sleep(self):
        """Induce transient unavailability"""
        pass

    def check_and_update_state_avl(self):
        if hasattr(self, "cm") and self.cm is not None:
            if len(self.state_avl_event_ts) > 0:
                next_event_ts = self.trainer_start_ts + (
                    self.state_avl_event_ts[0][0] / self.speedup_factor
                )
                if time.time() >= next_event_ts:
                    state_to_set = self.state_avl_event_ts.pop(0)[1]
                    old_status = self.avl_state.value
                    try:
                        self.avl_state = TrainerAvailState(state_to_set)
                    except ValueError:
                        logger.error(
                            f"Invalid status encountered: {state_to_set}. Retaining old status {old_status}."
                        )
                        return
                    new_status = self.avl_state.value
                    logger.info(
                        f"Changed the availability status of trainer {self.trainer_id} from {old_status} to {new_status}"
                    )
                    if self.client_notify["enabled"] == "True":
                        self._perform_channel_state_update(
                            tag="upload",
                            state=self.avl_state,
                            timestamp=str(time.time()),
                        )
            else:
                logger.debug(
                    f"No availability events pending for trainer {self.trainer_id}"
                )
        else:
            logger.info(
                f"Channel manager not set yet for trainer {self.trainer_id}. "
                f"Skipping avail status update. "
                f"Sleep for 20s before checking again."
            )
            time.sleep(20)

    def initialize(self) -> None:
        """Initialize role."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Net().to(self.device)
        logger.debug(
            f"Task_id: {self.trainer_id} initialize completed at timestamp: "
            f"{time.time()}"
        )

    def load_data(self) -> None:
        """Load data."""
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        dataset = CIFAR10(
            "/home/dgarg39/flame/lib/python/examples/async_cifar10/data",
            train=True,
            download=True,
            transform=transform_train,
        )

        # create indices into a list and convert to tensor
        indices = torch.tensor(self.trainer_indices_list)

        dataset = data_utils.Subset(dataset, indices)
        train_kwargs = {
            "batch_size": self.batch_size,
            "drop_last": True,
            "shuffle": True,
            "num_workers": 2,
        }

        self.train_loader = torch.utils.data.DataLoader(dataset, **train_kwargs)

        # Release the memory of the full dataset
        del dataset
        gc.collect()

        logger.debug(
            f"Task_id: {self.trainer_id} load_data completed at timestamp: "
            f"{time.time()}"
        )

    def train(self) -> None:
        logger.info(f"Entered train method for {self.trainer_id}")
        if self.task_to_perform != "train":
            logger.info(f"Trainer {self.trainer_id} is not required to train")
            return
        # don't enter the if condition if the three_state_avl switch is off
        # if we are checking for three_state_avl - check if the mechanism is to wait or exit
        if self.avl_state != TrainerAvailState.AVL_TRAIN:
            if self.wait_until_next_avl == "True":
                logger.info(
                    f"Trainer id {self.trainer_id} is not available to train. Waiting for it to be available"
                )
                while self.avl_state != TrainerAvailState.AVL_TRAIN:
                    time.sleep(1)
            else:
                logger.info(
                    f"Trainer id {self.trainer_id} is not available to train. Exiting training."
                )
                return

        logger.info(f"Trainer {self.trainer_id} available to train")

        """Train a model."""
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        # reset stat utility for OORT
        self.reset_stat_utility()

        for epoch in range(1, self.epochs + 1):
            self._train_epoch(epoch)

        # save dataset size so that the info can be shared with
        # aggregator
        self.dataset_size = len(self.train_loader.dataset)

        # emulate delays in training (due to compute resource and/or
        # dataset size and/or network latency) if enabled
        if self.training_delay_enabled == "True":
            time.sleep(self.training_delay_s / self.speedup_factor)
            logger.debug(
                f"Delayed training time for trainer "
                f"{self.trainer_id} by {self.training_delay_s}s"
            )

    def _train_epoch(self, epoch):
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
                loss = self.oort_loss(output, target.squeeze(), epoch, batch_idx)

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

        # normalize statistical utility of a trainer based on the size
        # of the dataset
        self.normalize_stat_utility(epoch)

    def evaluate(self) -> None:
        """Evaluate a model."""
        # Implement only forward pass evaluate if the trainer is available to train or to evaluate
        # Evaluate after train is written in the train_epoch method itself

        # Evaluate will be skipped if one of these three is satisfied:
        # 1. task_to_perform is train
        # 2. switch to check for three_state_avl is off
        # 3. Trainer is unavailable and we don't want it to wait for availability
        if (
            self.task_to_perform != "eval"
            or self.client_notify["trace"] == "two_state"
            or (
                self.avl_state == TrainerAvailState.UN_AVL
                and self.wait_until_next_avl == "False"
            )
        ):
            logger.warning(
                f"Evaluate (forward pass) will not be run for trainer id {self.trainer_id}. task_to_perform = {self.task_to_perform} and trainer avl_state = {self.avl_state.value} and wait_until_next_avl = {self.wait_until_next_avl}"
            )
            return

        if self.avl_state == TrainerAvailState.UN_AVL:
            logger.warning(
                f"Trainer id {self.trainer_id} is not available to perform forward pass evaluate. Waiting for it to be available"
            )
            while self.avl_state == TrainerAvailState.UN_AVL:
                time.sleep(1)

        logger.info(f"Starting eval (forward pass) for trainer id {self.trainer_id}")
        for epoch in range(1, self.epochs + 1):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                if self.use_oort_loss_fn == "False":
                    # Loss function to use with Fedbuff
                    loss = F.nll_loss(output, target)
                elif self.use_oort_loss_fn == "True":
                    # Calculate statistical utility of a trainer while
                    # calculating loss
                    loss = self.oort_loss(output, target.squeeze(), epoch, batch_idx)
                if batch_idx % 100 == 0:
                    done = batch_idx * len(data)
                    total = len(self.train_loader.dataset)
                    percent = 100.0 * batch_idx / len(self.train_loader)
                    logger.info(
                        f"epoch: {epoch} [{done}/{total} ({percent:.0f}%)]"
                        f"\tloss: {loss.item():.6f}"
                    )

            # normalize statistical utility of a trainer based on the size
            # of the dataset
            self.normalize_stat_utility(epoch)
        if self.training_delay_enabled == "True":
            # Updated eval duration to be one-third of training
            # duration since it is evidenced on text and through
            # profiling
            # Eval is 3X faster than training on CPU
            # Eval is 10-50X faster than training on CPUs due to NPUs
            # not supporting training. We take 20X
            eval_delay = math.floor(self.training_delay_s / 20.0)
            time.sleep(eval_delay / self.speedup_factor)
            logger.debug(
                f"Delayed eval time for trainer " f"{self.trainer_id} by {eval_delay}s"
            )

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
            time.sleep(1)  # Will check every 1 second
            self.check_and_update_state_avl()


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", type=str, default="./config.json", 
                        help="Path to config JSON file", required=False)
    parser.add_argument("--config-json", type=str, 
                        help="Config as JSON string (alternative to --config file)",
                        required=False)

    # Add a parser argument to get battery threshold (either 50 or 75)
    parser.add_argument(
        "--battery_threshold",
        type=int,
        choices=[50, 75],
        default=50,
        help="Battery threshold for the trainer 3-state events (either 50 or 75)",
        required=False,
    )

    # Add argument to speed up client's timescale by a factor
    parser.add_argument(
        "--speedup_factor",
        type=float,
        default=1.0,
        help="Speedup factor to accelarate all events and training/ eval durations from the trainer. Default- no acceleration",
        required=False,
    )

    args = parser.parse_args()
    
    # Handle config loading: either from file or JSON string
    if args.config_json:
        # Load config from JSON string (new programmatic spawning mode)
        config_dict = json.loads(args.config_json)
        # Create a temporary config file or pass dict directly
        # For now, write to temp file for compatibility with Config class
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_dict, f)
            temp_config_path = f.name
        
        try:
            config = Config(temp_config_path)
        finally:
            # Clean up temp file even if Config() fails
            os.unlink(temp_config_path)
    elif args.config:
        # Load config from file (legacy mode)
        config = Config(args.config)
    else:
        raise ValueError("Must provide either --config or --config-json")

    t = PyTorchCifar10Trainer(config, args.battery_threshold, args.speedup_factor)
    print(
        f"# Trainer id: {t.trainer_id}, has heartbeats_enabled: "
        f"{t.heartbeats_enabled}, has client_notify: "
        f"{t.client_notify['enabled']}, has "
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
    elif t.client_notify["trace"] is not None:
        logger.info(
            f"Will initiate thread to update state of " f"trainer {t.trainer_id}"
        )
        if t.client_notify["enabled"] == "True":
            logger.info(f"Will send avail notifications for trainer {t.trainer_id}")
        # Note that even though trainer sends notifications, only
        # async_oort will use it. Other selectors will not use it so
        # it can remain enabled.
        avail_notify_thread = threading.Thread(target=t.notify_trainer_avail)
        avail_notify_thread.daemon = True
        avail_notify_thread.start()

    t.compose()
    t.run()


if __name__ == "__main__":
    main()
