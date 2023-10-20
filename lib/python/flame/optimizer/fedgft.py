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
"""FedGFT optimizer."""
import logging

from flame.common.util import MLFramework, get_ml_framework_in_use
from flame.optimizer.bias import Bias
from flame.optimizer.fedavg import FedAvg
from flame.optimizer.regularizer.fedgft import FedGFTRegularizer

logger = logging.getLogger(__name__)


class FedGFT(FedAvg):
    """FedGFT class."""

    def __init__(self, fair, gamma, reg="l2"):
        """Initialize FedGFT optimizer instance."""
        self.agg_weights = None

        ml_framework_in_use = get_ml_framework_in_use()
        if ml_framework_in_use == MLFramework.PYTORCH:
            self.aggregate_fn = self._aggregate_pytorch
        else:
            raise NotImplementedError(
                "supported ml framework not found; "
                f"supported frameworks are: {[MLFramework.PYTORCH.name.lower()]}"
            )

        # type of fairness to optimize for
        self.fair = fair
        # corresponding regularizer
        self.regularizer = FedGFTRegularizer(fair, gamma, reg)

        # variable for tracking bias values
        self.bias = Bias(fair=self.fair, local=False)

    def update_bias(self, dataset_sizes, local_biases):
        self.bias.update_bias(dataset_sizes=dataset_sizes, local_biases=local_biases)

        # could store in metric collector
        logger.debug(f"current bias is {self.bias.val}")

    def get_bias(self):
        return self.bias.val
