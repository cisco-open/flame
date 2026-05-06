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
    HEARTBEAT = 5  # heartbeat from trainer to aggregator

    # a digest of all the workers in distributed learning
    MEMBER_DIGEST = 6
    RING_WEIGHTS = 7  # global model weights in distributed learning
    NEW_TRAINER = 8  # sending message for the arrival of a new trainer

    # a variable to indicate that a trainer is responsible to send weights
    # to a new trainer joining a distributed learning job
    IS_COMMITTER = 9

    MODEL_VERSION = 10  # model version used; an non-negative integer

    STAT_UTILITY = 11  # measured utility of a trainer based on Oort

    COORDINATED_ENDS = 12  # ends coordinated by a coordinator

    DATASAMPLER_METADATA = 13  # datasampler metadata

    META_INFO_REQ = 14  # Request for meta information
    META_INFO_RES = 15  # Response that contains meta information

    ALPHA_ADPT = 16  # adaptive hyperparameter used in FedDyn implementation

    REQ_COORDINATED_ENDS = 17  # request ends coordinated by a coordinator
    RES_COORDINATED_ENDS = 18  # get ends coordinated by a coordinator

    HYBRID_METADATA = 19  # metadata for hybrid aggregation

    BIAS = 20  # bias matrix used in FedGFT

    CONTROL_WEIGHTS = 21  # control variate weights used in SCAFFOLD
    CLIENT_WEIGHT = 22  # client-side control variate weight used in SCAFFOLD

    TASK_TO_PERFORM = 23  # assign between train and eval task
    AVL_STATE = 24  # tracks the states of trainers

    GRADIENTS = 25  # for paradigms that need to send gradients instead of weights
    VAR = 26  # variance of a group of updates
    DATA_ID = 27  # tracks specific data id to train on
    TOTAL_DATA_BINS = 28  # tracks per-client max data bins
    GRADIENTS_FOR_VAR_CHECK = (
        29  # for sending only the gradients to be used for var check
    )
    GRAD_POOL = 30  # stores all gradients until var isn't good enough
    ITERATION_PER_DATA_ID = 31  # stores round within a data id
    JVP_FOR_SNR_CHECK = 32
