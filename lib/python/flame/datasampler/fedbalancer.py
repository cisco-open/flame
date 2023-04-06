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
"""FedBalancerDataSampler class."""

import logging
import sys
import math
import numpy as np
from typing import Any

from flame.channel import Channel
from flame.common.util import MLFramework, get_ml_framework_in_use
from flame.datasampler import AbstractDataSampler

logger = logging.getLogger(__name__)

PROP_ROUND_START_TIME = "round_start_time"
PROP_ROUND_END_TIME = "round_end_time"


class FedBalancerDataSampler(AbstractDataSampler):
    def __init__(self, **kwargs) -> None:
        self.trainer_data_sampler = (
            FedBalancerDataSampler.FedBalancerTrainerDataSampler(**kwargs)
        )
        self.aggregator_data_sampler = (
            FedBalancerDataSampler.FedBalancerAggregatorDataSampler(**kwargs)
        )

    class FedBalancerTrainerDataSampler(AbstractDataSampler.AbstractTrainerDataSampler):
        """A trainer-side datasampler class based on FedBalancer."""

        def __init__(self, **kwargs):
            """Initailize instance."""
            super().__init__(**kwargs)

            ml_framework_in_use = get_ml_framework_in_use()
            if ml_framework_in_use != MLFramework.PYTORCH:
                raise NotImplementedError(
                    "FedBalancer is currently only implemented in PyTorch;"
                )

            import torch as T

            self.torch = T

            try:
                # Fedbalancer parameters

                # window size for round benefit calculation
                self._w = kwargs["w"]
                # loss threshold step size
                self._lss = kwargs["lss"]
                # deadline step size
                self._dss = kwargs["dss"]
                # portion to sample from over threshold group
                self._p = kwargs["p"]
                # noise factor for differential privacy
                self._noise_factor = kwargs["noise_factor"]

                # General training hyperparameters

                self.epochs = kwargs["epochs"]
                self.batch_size = kwargs["batchSize"]

            except KeyError:
                raise KeyError(
                    "one of the parameters among {w, lss, dss, p, noise_factor} "
                    + "is not specified in config,\nrecommended set of parameters"
                    + "are {20, 0.05, 0.05, 1.0, 0.0}"
                )

            self._loss_threshold = 0
            # Terminating a round with deadline is required in FedBalancer,
            # but not yet implemented in the current implementation.
            self._deadline = sys.maxsize

            self._stat_utility = 0
            self._ot_count = 0
            self._is_first_selected_round = True

            self._selected_loss_sum = 0
            self._selected_len = 0

            self._sample_loss_list = []
            self._train_duration_per_batch_list = []
            self._round_train_batch_count = 0

        def sample(self, dataset: Any, **kwargs) -> Any:
            """Select sample from the given dataset following FedBalancer algorithm."""
            logger.debug("calling fedbalancer datasampler")

            # calculate loss of train data samples only on its first selected round
            if self._is_first_selected_round:
                self._build_fb_sample_loss_dict(
                    dataset, kwargs["loss_fn"], kwargs["model"], kwargs["device"]
                )

            # calculate max trainable size of a trainer for epochs at current deadline
            max_trainable_size = 0
            if not self._is_first_selected_round:
                max_trainable_size = self._calculate_max_trainable_size()

            # if a client is fast enough to train its whole data for epochs within the
            # deadline, select all data
            if not self._is_first_selected_round and max_trainable_size > len(dataset):
                sampled_indices = list(range(len(dataset)))
                sampled_dataset = dataset
            else:
                # sample indexes with underthreshold loss (ut) and overthreshold loss (ot)
                ut_indices, ot_indices = [], []

                # read through the sample loss list and parse it in ut or ot indices lists,
                # based on the loss threshold value
                for idx, item in enumerate(self._sample_loss_list):
                    if item < self._loss_threshold:
                        ut_indices.append(idx)
                    else:
                        ot_indices.append(idx)

                # determine the number of samples to select from ut and ot group,
                # according to the fedbalancer paper
                sample_num = max(max_trainable_size, len(ot_indices))
                ut_sample_len = int(sample_num * (1 - self._p))
                ot_sample_len = sample_num - ut_sample_len

                # random sample from ut and ot without replacement
                sampled_ut_indices = np.random.choice(ut_indices, ut_sample_len, False)
                sampled_ot_indices = np.random.choice(ot_indices, ot_sample_len, False)

                sampled_indices = [*sampled_ut_indices, *sampled_ot_indices]

                sampled_dataset = self.torch.utils.data.Subset(dataset, sampled_indices)

            # calculate selected samples' loss sum and len as fedbalancer metadata,
            # that is being sent back to the aggregator for the calculation of
            # a new loss_threshold
            self._selected_loss_sum = 0

            for sample_index in sampled_indices:
                self._selected_loss_sum += self._sample_loss_list[sample_index]

            self._selected_len = len(sampled_indices)

            # reset number of trained batch count at current round with new sampled dataset
            self._round_train_batch_count = 0

            if self._is_first_selected_round:
                self._is_first_selected_round = False

            return sampled_dataset

        def load_dataset(self, dataset: Any) -> Any:
            """Change dataset instance to return index with each sample."""
            from flame.datasampler.util import PyTorchDatasetWithIndex

            return PyTorchDatasetWithIndex(dataset)

        def get_metadata(self) -> dict[str, Any]:
            """
            Returns the metadata of trainers of fedbalancer to send to
            aggregator for new loss_threshold and deadline calculation
            """
            # measure differentially-private statisticcs of sample loss values as metadata
            loss_low, loss_high = self._measure_DP_stats()

            # measure train duration per epoch for the current round
            epoch_train_duration = int(
                self._selected_len / self.batch_size + 1
            ) * np.mean(
                self._train_duration_per_batch_list[-self._round_train_batch_count :]
            )

            return {
                "loss_low": loss_low,
                "loss_high": loss_high,
                "selected_loss_sum": self._selected_loss_sum,
                "selected_len": self._selected_len,
                "epoch_train_duration": epoch_train_duration,
            }

        def handle_metadata_from_aggregator(self, metadata: dict[str, Any]) -> None:
            """
            Handle received metadata from aggregator.
            """
            self._loss_threshold = metadata["loss_threshold"]
            self._deadline = metadata["deadline"]

        def _build_fb_sample_loss_dict(
            self,
            dataset: Any,
            loss_fn: Any,
            model: Any,
            device: str,
        ) -> None:
            """
            Calculate loss of train data samples of a trainer
            on its first selected round.
            """

            # initialize sample loss list with a pair of index and loss value as zero
            self._sample_loss_list = [0 for _ in range(len(dataset))]
            full_samples_loader = self.torch.utils.data.DataLoader(
                dataset, batch_size=len(dataset)
            )

            with self.torch.no_grad():
                for sample in full_samples_loader:
                    data, target = sample[0].to(device), sample[1].to(device)
                    indices = sample[2]

                    output = model(data)
                    loss_list = loss_fn(output, target, reduction="none")

                    for k, idx in enumerate(indices):
                        self._sample_loss_list[idx] = loss_list[k]

        def _calculate_max_trainable_size(self) -> int:
            """
            Calculate max trainable size of a trainer for epochs
            at current deadline.
            """
            num_of_trainable_batches = (
                self._deadline
                / (self.epochs * np.mean(self._train_duration_per_batch_list))
                - 1
            )
            max_trainable_size = int(num_of_trainable_batches * self.batch_size - 1)

            return max_trainable_size

        def fb_loss(
            self,
            loss_fn: Any,
            output: Any,
            target: Any,
            epoch: int,
            indices: list[int],
        ) -> Any:
            """
            Measure the loss of a trainer during training.
            The trainer's statistical utility is measured at epoch 1.
            """
            if epoch == 1:
                loss_list = loss_fn(output, target, reduction="none")
                loss = loss_list.mean()

                # update the loss at the self._sample_loss_list with values
                # that are acquired during training
                for k, loss in enumerate(loss_list):
                    self._sample_loss_list[indices[k]] = loss.item()

                # filter loss list to only calculate utility with loss values
                # that have bigger loss than the loss threshold
                loss_list = self.torch.where(
                    loss_list >= self._loss_threshold,
                    loss_list,
                    self.torch.zeros_like(loss_list),
                )

                # count the samples with overthreshold loss values for normalization
                self._ot_count += self.torch.count_nonzero(
                    loss_list >= self._loss_threshold
                ).item()

                self._stat_utility += self.torch.square(loss_list).sum()
            else:
                loss = loss_fn(output, target)

            return loss

        def normalize_stat_utility(self, epoch) -> None:
            """
            Normalize statistical utility of a trainer based on the size
            of the trainer's datset, at epoch 1.
            """
            if epoch == 1:
                self._stat_utility = self._ot_count * math.sqrt(
                    self._stat_utility / self._ot_count
                )
            else:
                return

        def reset_stat_utility(self) -> None:
            """Reset the trainer's statistical utility to zero."""
            self._stat_utility = 0
            self._ot_count = 0

        def update_train_duration_per_batch_list(self, time: float) -> None:
            """Append the train duration per batch to the list."""
            self._train_duration_per_batch_list.append(time)
            self._round_train_batch_count += 1

        def _measure_DP_stats(self) -> tuple[float, float]:
            """
            Measure the differentially-private statistics of sample loss values
            according to the FedBalancer paper
            """
            loss_list = []
            for loss in self._sample_loss_list:
                loss_list.append(loss)

            dp_noise_1 = np.random.normal(0, self._noise_factor, 1)[0]
            dp_noise_2 = np.random.normal(0, self._noise_factor, 1)[0]

            loss_low = np.min(loss_list) + dp_noise_1
            loss_high = np.percentile(loss_list, 80) + dp_noise_2

            return loss_low, loss_high

    class FedBalancerAggregatorDataSampler(
        AbstractDataSampler.AbstractAggregatorDataSampler
    ):
        """An aggregator-side datasampler class based on FedBalancer."""

        def __init__(self, **kwargs):
            """Initailize instance."""
            super().__init__(**kwargs)

            try:
                # Fedbalancer parameters

                # window size for round benefit calculation
                self._w = kwargs["w"]
                # loss threshold step size
                self._lss = kwargs["lss"]
                # deadline step size
                self._dss = kwargs["dss"]
                # portion to sample from over threshold group
                self._p = kwargs["p"]
                # noise factor for differential privacy
                self._noise_factor = kwargs["noise_factor"]

                # General training hyperparameters

                self.epochs = kwargs["epochs"]
                self.batch_size = kwargs["batchSize"]

            except KeyError:
                raise KeyError(
                    "one of the parameters among {w, lss, dss, p, noise_factor} "
                    + "is not specified in config,\nrecommended set of parameters"
                    + "are {20, 0.05, 0.05, 1.0, 0.0}"
                )

            self._loss_threshold = 0
            # Terminating a round with deadline is required in FedBalancer,
            # but not yet implemented in the current implementation.
            self._deadline = sys.maxsize

            self._ltr = 0.0
            self._ddlr = 1.0

            self._round_benefit_history = []
            self._round_duration_history = {}
            self._round_epoch_train_duration_history = {}
            self._round_communication_duration_history = {}

            self._curr_round_loss_low = []
            self._curr_round_loss_high = []
            self._curr_round_selected_loss_sum = []
            self._curr_round_selected_len = []

        def get_metadata(self, round: int, selected_ends: list[str]) -> dict[str, Any]:
            """
            Returns the metadata variables of aggregator of fedbalancer
            that are distributed on trainers for sample selection.
            To this end, update loss threshold ratio and deadline ratio based on the
            round benefit history, then update loss threshold and deadline.
            Also, reset aggregator variables for current round.
            """

            # skips update fb variables if first round
            if round >= 2:
                self._update_fb_ratio(round)
                self._update_fb_loss_threshold()
                self._update_fb_deadline(selected_ends)

                (
                    self._curr_round_loss_low,
                    self._curr_round_loss_high,
                    self._curr_round_selected_loss_sum,
                    self._curr_round_selected_len,
                ) = ([], [], [], [])

            metadata = {
                "loss_threshold": self._loss_threshold,
                "deadline": self._deadline,
            }

            return metadata

        def handle_metadata_from_trainer(
            self,
            metadata: dict[str, Any],
            end: str,
            channel: Channel,
        ) -> None:
            """
            Handles the metadata of fedbalancer from trainers for
            new loss_threshold and deadline calculation
            """

            # add loss related metadata to list
            self._curr_round_loss_low.append(metadata["loss_low"])
            self._curr_round_loss_high.append(metadata["loss_high"])
            self._curr_round_selected_loss_sum.append(metadata["selected_loss_sum"])
            self._curr_round_selected_len.append(metadata["selected_len"])

            # calculate the time duration of train and communication from curr round
            round_start_time_tup = channel.get_end_property(end, PROP_ROUND_START_TIME)
            round_end_time_tup = channel.get_end_property(end, PROP_ROUND_END_TIME)
            if round_start_time_tup[0] == round_end_time_tup[0]:
                round_duration = (
                    round_end_time_tup[1] - round_start_time_tup[1]
                ).total_seconds()
            elif end in self._round_duration_history.keys():
                round_duration = self._round_duration_history[-1]
            else:
                return

            round_epoch_train_duration = metadata["epoch_train_duration"]
            round_communication_duration = (
                round_duration - self.epochs * round_epoch_train_duration
            )

            # save duration information. Check if the history list exists for this end
            # and append the information at the list
            if end not in self._round_duration_history.keys():
                self._round_duration_history[end] = []
            if end not in self._round_epoch_train_duration_history.keys():
                self._round_epoch_train_duration_history[end] = []
            if end not in self._round_communication_duration_history.keys():
                self._round_communication_duration_history[end] = []

            self._round_duration_history[end].append(round_duration)
            self._round_epoch_train_duration_history[end].append(
                round_epoch_train_duration
            )
            self._round_communication_duration_history[end].append(
                round_communication_duration
            )

            # we maintain the duration history of last 5 times which an end participated
            # at a round, according to the implementation of FedBalancer authors:
            # https://github.com/jaemin-shin/FedBalancer,
            # Commit 1d187c88de9b5b43e28c988b2423e9f616c80610
            self._round_duration_history[end] = self._round_duration_history[end][-5:]
            self._round_epoch_train_duration_history[
                end
            ] = self._round_epoch_train_duration_history[end][-5:]
            self._round_communication_duration_history[
                end
            ] = self._round_communication_duration_history[end][-5:]

        def _update_fb_ratio(self, round: int) -> None:
            """
            Update loss threshold ratio and deadline ratio based on the
            round benefit history.
            """

            # calculate the benefit of a round on the current configuration
            # with loss threshold and the deadline, then add it to history list
            curr_round_loss_sum = sum(self._curr_round_selected_loss_sum)
            curr_round_selected_len = sum(self._curr_round_selected_len)

            curr_round_benefit = (
                curr_round_loss_sum / curr_round_selected_len
            ) / self._deadline

            self._round_benefit_history.append(curr_round_benefit)

            lss = self._lss
            dss = self._dss

            # use round - 1 here as curr_round loss sum and selected len info
            # are from last round
            if (round - 1) % self._w == self._w - 1:
                if len(self._round_benefit_history) >= 2 * self._w:
                    last_window_benefit = np.mean(
                        self._round_benefit_history[-(self._w * 2) : -(self._w)]
                    )
                    curr_window_benefit = np.mean(
                        self._round_benefit_history[-(self._w) :]
                    )

                    if last_window_benefit - curr_window_benefit > 0:
                        self._ltr = min(self._ltr + lss, 1.0)
                        self._ddlr = max(self._ddlr - dss, 0.0)
                    else:
                        self._ltr = max(self._ltr - lss, 0.0)
                        self._ddlr = min(self._ddlr + dss, 1.0)

                    logger.debug(f"{self._ltr=}, {self._ddlr=}")

        def _update_fb_loss_threshold(self) -> None:
            """
            Update fedbalancer loss threshold based on the received
            loss_low values and loss_high values from a round.
            """

            # is loss_threshold_ratio (ltr) is 0 and loss_threshold is 0, it means
            # the training is in early stage, so leave loss_threshold as 0
            if self._loss_threshold == 0 and self._ltr == 0:
                logger.debug(f"{self._loss_threshold=}")
                return

            loss_low = np.min(self._curr_round_loss_low)
            loss_high = np.mean(self._curr_round_loss_high)

            self._loss_threshold = loss_low + (loss_high - loss_low) * self._ltr

            logger.debug(f"{loss_low=}, {loss_high=}, {self._loss_threshold=}")

        def _update_fb_deadline(self, selected_ends: list[str]) -> None:
            """
            Update fedbalancer deadline based on the round duration times of
            trianers from previous rounds.
            """
            deadline_low = self._find_peak_DDLE(selected_ends, 1)
            deadline_high = self._find_peak_DDLE(selected_ends, self.epochs)

            self._deadline = deadline_low + (deadline_high - deadline_low) * self._ddlr

            logger.debug(f"{deadline_low=}, {deadline_high=}, {self._deadline=}")

        def _find_peak_DDLE(self, selected_ends: list[str], epoch: int) -> int:
            """
            Search for deadline that results in max DDL-E (deadline efficiency) value
            when trainers (ends) train for specified number of epoch, as in fedbalancer.
            """
            max_DDLE_value = -1
            max_DDLE_time = -1

            expected_end_complete_time = {}

            for end in selected_ends:
                if end in self._round_communication_duration_history.keys():
                    expected_end_complete_time[end] = (
                        np.mean(self._round_communication_duration_history[end])
                        + np.mean(self._round_epoch_train_duration_history[end]) * epoch
                    )

            for candidate_deadline in range(1, sys.maxsize):
                round_complete_end_count = 0
                for end in expected_end_complete_time.keys():
                    if expected_end_complete_time[end] <= candidate_deadline:
                        round_complete_end_count += 1

                curr_DDLE_value = round_complete_end_count / candidate_deadline

                if max_DDLE_value < curr_DDLE_value:
                    max_DDLE_value = curr_DDLE_value
                    max_DDLE_time = candidate_deadline

                if round_complete_end_count == len(expected_end_complete_time.keys()):
                    break

            return max_DDLE_time
