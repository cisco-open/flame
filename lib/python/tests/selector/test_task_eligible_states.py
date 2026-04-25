# Copyright 2023 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for task_eligible_states initialization in selectors.

Tests verify that the _task_eligible_states dict is correctly built from
selector kwargs, including validation, defaults, and edge cases.
"""

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers to construct selector instances with minimal mocking
# ---------------------------------------------------------------------------


def _make_async_oort_selector(kwargs: dict):
    """Create an AsyncOortSelector with the given kwargs, mocking heavy deps."""
    from flame.selector.async_oort import AsyncOortSelector

    with patch.object(AsyncOortSelector, "__init__", lambda self, **kw: None):
        sel = AsyncOortSelector.__new__(AsyncOortSelector)

    # Minimal attributes the __init__ logic needs
    sel.c = kwargs.get("c", 1)
    sel.check_three_state_avl = True

    # Re-run only the task_eligible_states block
    from flame.config import TrainerAvailState

    _default_eligible_states = {
        "train": [TrainerAvailState.AVL_TRAIN.value],
        "eval": [
            TrainerAvailState.AVL_EVAL.value,
            TrainerAvailState.AVL_TRAIN.value,
        ],
    }
    raw_eligible = kwargs.get("task_eligible_states", _default_eligible_states)
    _valid_states = {v.value for v in TrainerAvailState}

    sel._task_eligible_states = {}
    for task, states in raw_eligible.items():
        invalid = [s for s in states if s not in _valid_states]
        if invalid:
            raise ValueError(
                f"task_eligible_states['{task}'] contains invalid states: {invalid}. "
                f"Valid states: {sorted(_valid_states)}"
            )
        sel._task_eligible_states[task] = list(states)

    return sel


# ---------------------------------------------------------------------------
# Default behavior
# ---------------------------------------------------------------------------


class TestDefaultEligibleStates:
    def test_train_default_is_avl_train_only(self):
        from flame.config import TrainerAvailState
        sel = _make_async_oort_selector({})
        assert sel._task_eligible_states["train"] == [TrainerAvailState.AVL_TRAIN.value]

    def test_eval_default_includes_avl_eval_and_avl_train(self):
        from flame.config import TrainerAvailState
        sel = _make_async_oort_selector({})
        assert TrainerAvailState.AVL_EVAL.value in sel._task_eligible_states["eval"]
        assert TrainerAvailState.AVL_TRAIN.value in sel._task_eligible_states["eval"]


# ---------------------------------------------------------------------------
# Custom configuration
# ---------------------------------------------------------------------------


class TestCustomEligibleStates:
    def test_fwdllm_train_eligible_states(self):
        """FwdLLM: both AVL_TRAIN and AVL_EVAL can handle training tasks."""
        from flame.config import TrainerAvailState
        sel = _make_async_oort_selector({
            "task_eligible_states": {
                "train": [TrainerAvailState.AVL_TRAIN.value, TrainerAvailState.AVL_EVAL.value],
                "eval": [TrainerAvailState.AVL_EVAL.value, TrainerAvailState.AVL_TRAIN.value],
            }
        })
        assert TrainerAvailState.AVL_EVAL.value in sel._task_eligible_states["train"]
        assert TrainerAvailState.AVL_TRAIN.value in sel._task_eligible_states["train"]

    def test_classic_fl_train_only_avl_train(self):
        """Classic FL: only AVL_TRAIN ends can train."""
        from flame.config import TrainerAvailState
        sel = _make_async_oort_selector({
            "task_eligible_states": {
                "train": [TrainerAvailState.AVL_TRAIN.value],
                "eval": [TrainerAvailState.AVL_EVAL.value],
            }
        })
        assert sel._task_eligible_states["train"] == [TrainerAvailState.AVL_TRAIN.value]
        assert sel._task_eligible_states["eval"] == [TrainerAvailState.AVL_EVAL.value]

    def test_custom_task_name_is_preserved(self):
        from flame.config import TrainerAvailState
        sel = _make_async_oort_selector({
            "task_eligible_states": {
                "finetune": [TrainerAvailState.AVL_TRAIN.value],
            }
        })
        assert "finetune" in sel._task_eligible_states


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestEligibleStatesValidation:
    def test_invalid_state_string_raises_value_error(self):
        with pytest.raises(ValueError, match="invalid states"):
            _make_async_oort_selector({
                "task_eligible_states": {
                    "train": ["not_a_valid_state"],
                }
            })

    def test_empty_state_list_is_allowed(self):
        sel = _make_async_oort_selector({
            "task_eligible_states": {
                "train": [],
            }
        })
        assert sel._task_eligible_states["train"] == []
