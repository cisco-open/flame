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

from flame.common.typing import ModelWeights
from flame.common.util import (
    MLFramework,
    get_ml_framework_in_use,
)


class DifferentialPrivacy:
    """
    Differential Privacy class that adds noise to the trainer's update weights.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize instance."""

        try:
            # clip threshold
            self._clip_threshold = kwargs["clip_threshold"]
            # noise factor for differential privacy
            self._noise_factor = kwargs["noise_factor"]

        except KeyError:
            raise KeyError(
                "one of the parameters among {clip_threshold, noise_factor} "
                + "is not specified in config"
            )

        framework = get_ml_framework_in_use()
        if framework == MLFramework.TENSORFLOW:
            raise NotImplementedError(
                "Differential privacy is currently only implemented in PyTorch;"
            )
        elif framework == MLFramework.PYTORCH:
            import torch as T

            self.torch = T
            self.apply_dp_fn = self._apply_dp_torch

    def _apply_dp_torch(self, delta_weights: ModelWeights) -> ModelWeights:
        """Apply dp noise to the delta_weights."""
        delta_weights = self._apply_clip_norm_torch(delta_weights, self._clip_threshold)
        return {
            k: d_w + self.torch.normal(mean=0, std=self._noise_factor, size=d_w.shape)
            for k, d_w in delta_weights.items()
        }

    def _apply_clip_norm_torch(
        self, delta_weights: ModelWeights, max_norm: float, norm_type: float = 2.0
    ) -> ModelWeights:
        """Clip grad norm of weights based on the max_norm value for better privacy."""

        total_norm = self.torch.norm(
            self.torch.stack(
                [
                    self.torch.norm(delta_weights[k], norm_type)
                    for k in delta_weights.keys()
                ]
            ),
            norm_type,
        )

        if total_norm > max_norm:
            delta_weights = {
                k: self.torch.mul(d_w, (max_norm / (total_norm + 1e-6)))
                for k, d_w in delta_weights.items()
            }

        return delta_weights
