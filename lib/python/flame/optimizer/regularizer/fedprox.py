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

from flame.common.util import get_params_as_vector_pytorch

from .default import Regularizer

logger = logging.getLogger(__name__)


class FedProxRegularizer(Regularizer):
    """Regularizer class."""

    def __init__(self, mu):
        """Initialize FedProxRegularizer instance."""
        super().__init__()
        self.mu = mu
        self.state_dict = dict()

    def get_term(self, **kwargs):
        """Calculate proximal term for client-side regularization."""
        import torch

        w = kwargs["w"]
        w_vector = get_params_as_vector_pytorch(w)

        if "w_t_vector" in self.state_dict:
            w_t_vector = self.state_dict["w_t_vector"]
        else:
            w_t = kwargs["w_t"]
            w_t_vector = get_params_as_vector_pytorch(w_t)
            self.state_dict["w_t_vector"] = w_t_vector

        return (self.mu / 2) * torch.sum(torch.pow(w_vector - w_t_vector, 2))

    def update(self):
        del self.state_dict["w_t_vector"]
