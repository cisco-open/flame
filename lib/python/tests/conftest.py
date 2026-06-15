# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Shared pytest fixtures for selector and availability tests."""

from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _force_pytorch_framework(monkeypatch):
    """Selectors gate on get_ml_framework_in_use() == PYTORCH at init.

    For pure-Python unit tests we don't need torch; pretend the framework
    is PyTorch so selector __init__ succeeds.
    """
    from flame.common.util import MLFramework
    import flame.common.util as _cu

    monkeypatch.setattr(_cu, "get_ml_framework_in_use", lambda: MLFramework.PYTORCH)
    for mod_name in (
        "flame.selector.oort",
        "flame.selector.refl_oort",
        "flame.selector.async_oort",
    ):
        try:
            mod = __import__(mod_name, fromlist=["get_ml_framework_in_use"])
        except ImportError:
            continue
        if hasattr(mod, "get_ml_framework_in_use"):
            monkeypatch.setattr(
                mod, "get_ml_framework_in_use", lambda: MLFramework.PYTORCH
            )
    yield


class FakeEnd:
    """Minimal End stand-in: just a property bag."""

    def __init__(self, end_id: str):
        self.end_id = end_id
        self._props: dict = {}

    def set_property(self, key, value):
        self._props[key] = value

    def get_property(self, key):
        return self._props.get(key)


@pytest.fixture
def make_end():
    """Factory: make_end('id', stat_utility=1.0, round_duration=timedelta(seconds=10))."""
    def _make(end_id: str, **props):
        e = FakeEnd(end_id)
        for k, v in props.items():
            e.set_property(k, v)
        return e
    return _make


@pytest.fixture
def make_ends(make_end):
    """Factory: make_ends(['a', 'b', 'c']) or make_ends(count=20, prefix='t')."""
    def _make(ids=None, *, count=None, prefix="end", **props):
        if ids is None:
            ids = [f"{prefix}{i}" for i in range(count)]
        return {eid: make_end(eid, **props) for eid in ids}
    return _make


@pytest.fixture
def channel_props():
    """Minimal channel_props dict with round counter."""
    return {"round": 1, "cur_time": 0.0}


@pytest.fixture
def fast_round_duration():
    return timedelta(seconds=10)
