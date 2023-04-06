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
"""DefaultDataSampler class."""

import logging
from typing import Any

from flame.channel import Channel
from flame.datasampler import AbstractDataSampler

logger = logging.getLogger(__name__)


class DefaultDataSampler(AbstractDataSampler):
    def __init__(self) -> None:
        self.trainer_data_sampler = DefaultDataSampler.DefaultTrainerDataSampler()
        self.aggregator_data_sampler = DefaultDataSampler.DefaultAggregatorDataSampler()

    class DefaultTrainerDataSampler(AbstractDataSampler.AbstractTrainerDataSampler):
        """A default trainer-side datasampler class."""

        def __init__(self, **kwargs):
            """Initailize instance."""
            super().__init__()

        def sample(self, dataset: Any, **kwargs) -> Any:
            """Return all dataset from the given dataset."""
            logger.debug("calling default datasampler")

            return dataset

        def load_dataset(self, dataset: Any) -> None:
            """Change dataset instance to return index with each sample."""
            return dataset

        def get_metadata(self) -> dict[str, Any]:
            """Return metadata to send to aggregator-side datasampler."""
            return {}

        def handle_metadata_from_aggregator(self, metadata: dict[str, Any]) -> None:
            """Handle aggregator metadata for datasampler."""
            pass

    class DefaultAggregatorDataSampler(
        AbstractDataSampler.AbstractAggregatorDataSampler
    ):
        """A default aggregator-side datasampler class."""

        def __init__(self, **kwargs):
            """Initailize instance."""
            super().__init__()

        def get_metadata(self, end: str, round: int) -> dict[str, Any]:
            """Return metadata to send to trainer-side datasampler."""
            return {}

        def handle_metadata_from_trainer(
            self,
            metadata: dict[str, Any],
            end: str,
            channel: Channel,
        ) -> None:
            """Handle trainer metadata for datasampler."""
            pass
