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
"""datasampler abstract class."""

from abc import ABC, abstractmethod
from typing import Any

from flame.channel import Channel


class AbstractDataSampler(ABC):
    class AbstractTrainerDataSampler(ABC):
        """Abstract base class for trainer-side datasampler implementation."""

        def __init__(self, **kwargs) -> None:
            """Initialize an instance with keyword-based arguments."""
            for key, value in kwargs.items():
                setattr(self, key, value)

        @abstractmethod
        def sample(self, dataset: Any, **kwargs) -> Any:
            """Abstract method to sample data.

            Parameters
            ----------
            dataset: Dataset of a trainer to select samples from
            kwargs: other arguments specific to each datasampler algorithm

            Returns
            -------
            dataset: Dataset that only contains selected samples
            """

        @abstractmethod
        def load_dataset(self, dataset: Any) -> Any:
            """Process dataset instance for datasampler."""

        @abstractmethod
        def get_metadata(self) -> dict[str, Any]:
            """Return metadata to send to aggregator-side datasampler."""

        @abstractmethod
        def handle_metadata_from_aggregator(self, metadata: dict[str, Any]) -> None:
            """Handle aggregator metadata for datasampler."""

    class AbstractAggregatorDataSampler(ABC):
        """Abstract base class for aggregator-side datasampler implementation."""

        def __init__(self, **kwargs) -> None:
            """Initialize an instance with keyword-based arguments."""
            for key, value in kwargs.items():
                setattr(self, key, value)

        @abstractmethod
        def get_metadata(self, end: str, round: int) -> dict[str, Any]:
            """Return metadata to send to trainer-side datasampler."""

        @abstractmethod
        def handle_metadata_from_trainer(
            self,
            metadata: dict[str, Any],
            end: str,
            channel: Channel,
        ) -> None:
            """Handle trainer metadata for datasampler."""
