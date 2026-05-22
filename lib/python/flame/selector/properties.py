# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""End-property string keys shared across selectors and top_aggregator."""

# Identity / bookkeeping
PROP_END_ID = "end_id"
PROP_SELECTED_COUNT = "selected_count"
PROP_UPDATE_COUNT = "update_count"
PROP_LAST_SELECTED_ROUND = "last_selected_round"
PROP_LAST_EVAL_ROUND = "last_eval_round"
PROP_LAST_ENGAGED_ROUND = "last_engaged_round"

# Round timing
PROP_ROUND_START_TIME = "round_start_time"
PROP_ROUND_END_TIME = "round_end_time"
PROP_ROUND_DURATION = "round_duration"

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
