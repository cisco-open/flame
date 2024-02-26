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
"""SCAFFOLD Regularizer."""
import logging

from collections import defaultdict
from copy import deepcopy

from flame.common.constants import TrainState
from flame.common.util import (get_params_as_vector_pytorch,
                               get_params_detached_pytorch)
from math import ceil
from flame.optimizer.regularizer.default import Regularizer

logger = logging.getLogger(__name__)


class ScaffoldRegularizer(Regularizer):
    """SCAFFOLD Regularizer class."""

    def __init__(self, k):
        """Initialize ScaffoldRegularizer instance."""
        super().__init__()

        # save K hyperparameter (number of batches used in one round of training)
        self.k = k

        # save states in dictionary
        self.state_dict = dict()

        # local c-terms are initialized to 0.0
        self.c_loc = defaultdict(float)
        self.c_loc_vector = 0.0

    def get_term(self, **kwargs):
        """Calculate extra terms for client-side regularization."""
        import torch

        curr_model = kwargs["curr_model"]

        w = [param for param in curr_model.parameters()]

        # concatenate parameters into a vector
        w_vector = get_params_as_vector_pytorch(w)

        # loss term
        loss_algo = torch.sum(w_vector * self.state_vector_diff)

        return loss_algo

    def save_state(self, state: TrainState, **kwargs):
        if state == TrainState.PRE:
            self.state_dict["glob_model"] = deepcopy(kwargs["glob_model"])
        elif state == TrainState.POST:
            self.state_dict["loc_model"] = deepcopy(kwargs["loc_model"])

    def set_c_glob(self, c_glob):
        self.c_glob = c_glob
        c_model = deepcopy(self.state_dict["glob_model"])
        c_model.load_state_dict(self.c_glob)
        c_glob_params = get_params_detached_pytorch(c_model)
        self.c_glob_vector = get_params_as_vector_pytorch(c_glob_params)

        # for regularizer
        self.state_vector_diff = (
            -self.c_loc_vector + self.c_glob_vector / self.client_weight
        )

    def update(self):
        """Update the c-terms."""
        loc_model = self.state_dict["loc_model"]
        glob_model = self.state_dict["glob_model"]
        w = get_params_detached_pytorch(loc_model)
        w_t = get_params_detached_pytorch(glob_model)

        # concatenate weights into a vector
        w_vector = get_params_as_vector_pytorch(w)
        w_t_vector = get_params_as_vector_pytorch(w_t)

        w_state_dict = loc_model.state_dict()
        w_t_state_dict = glob_model.state_dict()

        diff_mult = 1 / (self.k * self.learning_rate)

        # adjust local vector
        self.c_loc_vector = (
            self.c_loc_vector - self.c_glob_vector + diff_mult * (w_t_vector - w_vector)
        )

        # adjust local pytorch state_dict
        new_c = {
            k: self.c_loc[k]
            - self.c_glob[k]
            + diff_mult * (w_t_state_dict[k] - w_state_dict[k])
            for k in w_state_dict
        }

        self.delta_c_loc = {
            k: (new_c[k] - self.c_loc[k]) * self.client_weight for k in new_c
        }

        self.c_loc = new_c
