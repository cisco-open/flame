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
"""A class that contains train result and its meta data."""


class TrainResult(object):
    """TrainResult class."""

    def __init__(
        self,
        weights=None,
        count=0,
        version=0,
        stat_utility=0,
        completion_time=None,
        round_duration=None,
        staleness=0,
        end_id=None,
    ):
        """
        Initialize TrainResult.

        Args:
            weights: Model weights
            count: Number of samples used for training
            version: Version number of the model
            stat_utility: Statistical utility (e.g., loss value)
            completion_time: Timestamp when training completed (for REFL)
            round_duration: Duration of training round (for REFL)
            staleness: Number of rounds this update is stale (for REFL)
            end_id: Identifier of the trainer/end that produced this result
        """

        self.weights = weights
        self.count = count
        self.version = version
        self.stat_utility = stat_utility

        # REFL-specific fields
        self.completion_time = completion_time
        self.round_duration = round_duration
        self.staleness = staleness
        self.end_id = end_id
