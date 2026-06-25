# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""End-property string keys shared across selectors and top_aggregator."""

# Identity / bookkeeping
PROP_END_ID = "end_id"
PROP_SELECTED_COUNT = "selected_count"
PROP_UPDATE_COUNT = "update_count"
PROP_LAST_SELECTED_ROUND = "last_selected_round"
# Aggregator round at which this end's update was last RECEIVED. This is the
# reference Oort/REFL `time_stamp` (set to `self.epoch` at result-processing,
# initialized at registration) that the UCB temporal-uncertainty term divides by.
PROP_LAST_RETURNED_ROUND = "last_returned_round"
PROP_LAST_EVAL_ROUND = "last_eval_round"
PROP_LAST_ENGAGED_ROUND = "last_engaged_round"

# Round timing
PROP_ROUND_START_TIME = "round_start_time"
PROP_ROUND_END_TIME = "round_end_time"
PROP_CLIENT_TASK_TRAIN_DURATION = "client_task_train_duration_s"

# Simulated time (time_mode="simulated"): trainer-reported virtual completion
# time, used to order updates by a virtual clock and to source round duration.
PROP_SIM_SEND_TS = "sim_send_ts"
PROP_SIM_COMPLETION_TS = "sim_completion_ts"

# Training metadata
PROP_DATASET_SIZE = "dataset_size"
PROP_STAT_UTILITY = "stat_utility"      # I_m / Oort utility
PROP_LOCAL_ACCURACY = "local_accuracy"  # FedDance a_m
PROP_UTILITY = "utility"                # generic computed utility
PROP_TOTAL_UNAVAIL_DURATION = "total_unavail_duration"

# Availability
PROP_AVL_STATE = "avl_state"

# FedDance per-end computed values
PROP_LAMBDA = "lambda_m"
PROP_V = "v_m"
PROP_I = "i_m"
PROP_A = "a_m"
PROP_U = "u_m"
