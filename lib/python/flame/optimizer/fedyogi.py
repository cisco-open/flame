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

"""FedYogi optimizer"""
"""https://arxiv.org/abs/2003.00295"""
import logging

from .fedopt import FedOPT

logger = logging.getLogger(__name__)

class FedYogi(FedOPT):
    """FedYogi class."""
    logger.debug("Initializing fedyogi")

    def __init__(self, beta_1=0.9, beta_2=0.99, eta=1e-2, tau=1e-3):
        """Initialize FedYogi instance."""
        super().__init__(beta_1, beta_2, eta, tau)
        logger.debug("Initializing fedyogi")

    def _delta_v_pytorch(self):
        import torch
        self.v_t = {k: self.v_t[k] - (1 - self.beta_2) * self.d_t[k]**2 * torch.sign(self.v_t[k] - self.d_t[k]**2) for k in self.v_t.keys()}
        return

    def _delta_v_tensorflow(self):
        import tensorflow as tf
        self.v_t = [self.v_t[idx] - (1 - self.beta_2) * self.d_t[idx]**2 * tf.sign(self.v_t[idx] - self.d_t[idx]**2) for idx in range(len(self.v_t))]
        return
