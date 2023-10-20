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
"""FedGFT Bias class definition."""

import logging

from flame.common.util import MLFramework, get_ml_framework_in_use

logger = logging.getLogger(__name__)


class Bias:
    def __init__(self, fair="", local=True):
        ml_framework_in_use = get_ml_framework_in_use()

        # only support pytorch
        if ml_framework_in_use != MLFramework.PYTORCH:
            raise NotImplementedError(
                "supported ml framework not found; "
                f"supported frameworks (for fedgft) are: {[MLFramework.PYTORCH.name.lower()]}"
            )

        # type of fairness
        self.fair = fair

        # terms in group-based fairness equation (a/b - c/d)
        self.a = 0.0
        self.b = 0.0
        self.c = 0.0
        self.d = 0.0

        # bias value
        self.val = 0.0
        self.sign = 0.0

        # local/global bias object
        self.local = local

        # role-specific variable initialization
        if local:
            self.local_val = 0.0
            self.global_a = 1.0
            self.global_b = 1.0
            self.global_c = 1.0
            self.global_d = 1.0

    def update_bias(self, **kwargs):
        if self.local:
            global_bias = kwargs["global_bias"]

            self.global_a = global_bias.a
            self.global_b = global_bias.b
            self.global_c = global_bias.c
            self.global_d = global_bias.d
            self.val = global_bias.val
            self.sign = global_bias.sign
        else:
            dataset_sizes = kwargs["dataset_sizes"]
            local_biases = kwargs["local_biases"]

            num_sample = sum(dataset_sizes.values())

            # sync a, b, c, d
            self.a = sum(
                [
                    local_biases[key].a * dataset_sizes[key] / num_sample
                    for key in local_biases
                ]
            )
            self.b = sum(
                [
                    local_biases[key].b * dataset_sizes[key] / num_sample
                    for key in local_biases
                ]
            )
            self.c = sum(
                [
                    local_biases[key].c * dataset_sizes[key] / num_sample
                    for key in local_biases
                ]
            )
            self.d = sum(
                [
                    local_biases[key].d * dataset_sizes[key] / num_sample
                    for key in local_biases
                ]
            )

            if self.fair == "SP" or self.fair == "EOP":
                self.val = sum(
                    [
                        local_biases[key].val * dataset_sizes[key] / num_sample
                        for key in local_biases
                    ]
                )

            self.sign = 0.0 if self.val == 0 else (1.0 if self.val > 0 else -1.0)

    def calculate_bias_batch(self, output, target, group):
        """Calculate bias on output batch."""
        import torch

        if self.fair == "SP":
            a = torch.sum(output[:, 1] * (group == 0))
            c = torch.sum(output[:, 1] * (group == 1))
            b = torch.sum(group == 0)
            d = torch.sum(group == 1)
        elif self.fair == "EOP":
            a = torch.sum(output[:, 1] * ((group == 0) & (target == 1)))
            c = torch.sum(output[:, 1] * ((group == 1) & (target == 1)))
            b = torch.sum((group == 0) & (target == 1))
            d = torch.sum((group == 1) & (target == 1))
        elif self.fair == "CAL":
            a = torch.sum(output[:, 1] * ((group == 0) & (target == 1)))
            c = torch.sum(output[:, 1] * ((group == 1) & (target == 1)))
            b = torch.sum(output[:, 1] * (group == 0))
            d = torch.sum(output[:, 1] * (group == 1))
        else:
            raise ValueError("Fairness type not supported.")

        return a, b, c, d

    def calculate_bias(self, model, data_loader):
        """Calculate the bias over complete dataset."""
        import torch

        device = next(model.parameters()).device
        res = torch.zeros(4)
        for data, target, group in data_loader:
            data, target, group = (
                data.to(device),
                target.to(device),
                group.to(device),
            )
            output = torch.exp(model(data))
            res += torch.Tensor(self.calculate_bias_batch(output, target, group))

        return res / len(data_loader.dataset)

    def update_local_bias_params(self, model, train_loader):
        """Update all bias parameters stored in bias based on dataset and current model."""
        import torch

        model.train(False)
        with torch.no_grad():
            a, b, c, d = self.calculate_bias(model, train_loader)
            self.a = a.item()
            self.b = max(b.item(), 1e-8)
            self.c = c.item()
            self.d = max(d.item(), 1e-8)
            self.local_val = self.a / self.b - self.c / self.d
            self.val = self.a / self.global_b - self.c / self.global_d
