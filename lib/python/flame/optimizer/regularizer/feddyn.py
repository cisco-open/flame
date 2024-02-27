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
"""FedDyn Regularizer."""
import logging

from flame.common.constants import TrainState
from flame.common.util import get_params_as_vector_pytorch, get_params_detached_pytorch
from flame.optimizer.regularizer.default import Regularizer

logger = logging.getLogger(__name__)


class FedDynRegularizer(Regularizer):
    """FedDyn Regularizer class."""

    def __init__(self, alpha):
        """Initialize FedDynRegularizer instance."""
        super().__init__()
        self.alpha = alpha

        # save states in dictionary
        self.state_dict = dict()

        # prev_grad is initialized to 0.0 for simplicity of implementation
        # this later becomes a tensor of length equal to the number of parameters
        self.prev_grad = 0.0

    def get_term(self, **kwargs):
        """Calculate extra terms for client-side regularization."""
        import torch

        curr_model = kwargs["curr_model"]
        prev_model = kwargs["prev_model"]

        w = [param for param in curr_model.parameters()]

        # concatenate weights into a vector
        w_vector = get_params_as_vector_pytorch(w)

        if "w_t_vector" in self.state_dict:
            w_t_vector = self.state_dict["w_t_vector"]
        else:
            w_t = [param for param in prev_model.parameters()]
            w_t_vector = get_params_as_vector_pytorch(w_t)
            self.state_dict["w_t_vector"] = w_t_vector

        # weight decay term using alpha parameter
        w_decay_term = (self.alpha / 2) * torch.sum(torch.pow(w_vector, 2))

        # remaining loss term
        loss_algo = self.alpha * torch.sum(w_vector * (-w_t_vector + self.prev_grad))

        return loss_algo + w_decay_term

    def save_state(self, state: TrainState, **kwargs):
        if state == TrainState.PRE:
            self.state_dict["glob_model"] = get_params_detached_pytorch(
                kwargs["glob_model"]
            )
        elif state == TrainState.POST:
            self.state_dict["loc_model"] = get_params_detached_pytorch(
                kwargs["loc_model"]
            )

    def update(self):
        """Update the previous gradient."""
        w = self.state_dict["loc_model"]
        w_t = self.state_dict["glob_model"]

        # concatenate weights into a vector
        w_vector = get_params_as_vector_pytorch(w)
        w_t_vector = get_params_as_vector_pytorch(w_t)

        # adjust prev_grad
        self.prev_grad += w_vector - w_t_vector

        del self.state_dict["w_t_vector"]
