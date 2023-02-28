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
"""FedProx Regularizer."""
import logging

from .default import Regularizer

logger = logging.getLogger(__name__)


class FedProxRegularizer(Regularizer):
    """Regularizer class."""

    def __init__(self, mu):
        """Initialize FedProxRegularizer instance."""
        super().__init__()
        self.mu = mu

    def get_term(self, **kwargs):
        """Calculate proximal term for client-side regularization."""
        import torch
        w = kwargs['w']
        w_t = kwargs['w_t']
        norm_sq = 0.0
        for loc_param, glob_param in zip(w, w_t):
            norm_sq += torch.sum(torch.pow(loc_param - glob_param, 2))
        return (self.mu / 2) * norm_sq
