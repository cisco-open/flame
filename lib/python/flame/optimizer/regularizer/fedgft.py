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
"""FedGFT Regularizer."""
import logging

from flame.common.util import MLFramework, get_ml_framework_in_use
from flame.optimizer.bias import Bias
from flame.optimizer.regularizer.default import Regularizer

logger = logging.getLogger(__name__)


class FedGFTRegularizer(Regularizer):
    """FedGFTRegularizer class."""

    def __init__(self, fair, gamma, reg="l2"):
        """Initialize FedGFTRegularizer instance."""
        ml_framework_in_use = get_ml_framework_in_use()

        # only support pytorch
        if ml_framework_in_use != MLFramework.PYTORCH:
            raise NotImplementedError(
                "supported ml framework not found; "
                f"supported frameworks (for fedgft) are: {[MLFramework.PYTORCH.name.lower()]}"
            )

        super().__init__()
        # type of fairness to optimize for
        self.fair = fair
        # penalty parameter in [0,inf)
        self.gamma = gamma
        # regularization function type for bias term
        self.reg = reg

        # variable for tracking bias values
        self.bias = Bias(fair=self.fair, local=True)

    def get_term(self, **kwargs):
        """Regularizer term for bias optimization."""
        import torch

        gamma = abs(self.bias.val * 2) if self.gamma == "auto" else self.gamma
        output = kwargs["output"]
        target = kwargs["target"]
        group = kwargs["group"]
        if self.fair:
            a, b, c, d = self.bias.calculate_bias_batch(
                torch.exp(output), target, group
            )
            if self.fair == "SP" or self.fair == "EOP":
                res = (a / self.bias.global_b - c / self.bias.global_d) / len(target)
            elif self.fair == "CAL":
                res = -(a / self.bias.global_b - c / self.bias.global_d) / len(target)
            else:
                raise ValueError("Fairness type not supported")
        else:
            res = 0.0

        coef = self.bias.sign if self.reg == "id" else self.bias.val
        return coef * gamma * res

    def update_local_bias_params(self, model, train_loader):
        """Update all bias parameters stored in Bias object based on dataset and current model."""
        self.bias.update_local_bias_params(model, train_loader)

    def update_bias(self, global_bias):
        """Update bias parameters stored in Bias object based on global bias."""
        self.bias.update_bias(global_bias=global_bias)

    def get_local_bias(self):
        return self.bias.local_val
