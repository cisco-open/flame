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
"""Message class."""

from enum import Enum


class MessageType(Enum):
    """Define Message types."""

    WEIGHTS = 1  # model weights
    EOT = 2  # end of training
    DATASET_SIZE = 3  # dataset size
    ROUND = 4  # round number

    # a digest of all the workers in distributed learning
    MEMBER_DIGEST = 5
    RING_WEIGHTS = 6  # global model weights in distributed learning
    NEW_TRAINER = 7  # sending message for the arrival of a new trainer

    # a variable to indicate that a trainer is responsible to send weights
    # to a new trainer joining a distributed learning job
    IS_COMMITTER = 8

    MODEL_VERSION = 9  # model version used; an non-negative integer

    STAT_UTILITY = 10  # measured utility of a trainer based on Oort

    COORDINATED_ENDS = 11  # ends coordinated by a coordinator

    DATASAMPLER_METADATA = 12  # datasampler metadata

    META_INFO_REQ = 13  # Request for meta information
    META_INFO_RES = 14  # Response that contains meta information

    ALPHA_ADPT = 15 # adaptive hyperparameter used in FedDyn implementation

