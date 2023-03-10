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
"""Config parser."""

import json
import typing as t
from enum import Enum

from pydantic import BaseModel as pydBaseModel
from pydantic import Field


class FlameSchema(pydBaseModel):
    pass


GROUPBY_DEFAULT_GROUP = "default"
GROUP_ASSOCIATION_SEPARATOR = "/"
DEFAULT_HYPERARAMETERS_DICT = {"rounds": 1, "epochs": 1, "batchSize": 16}


class BackendType(str, Enum):
    """Define backend types."""

    LOCAL = "local"
    P2P = "p2p"
    MQTT = "mqtt"


class RegistryType(str, Enum):
    """Define model registry types."""

    DUMMY = "dummy"
    MLFLOW = "mlflow"


class OptimizerType(str, Enum):
    """Define optimizer types."""

    FEDAVG = "fedavg"
    FEDADAGRAD = "fedadagrad"
    FEDADAM = "fedadam"
    FEDYOGI = "fedyogi"
    # FedBuff from https://arxiv.org/pdf/1903.03934.pdf and
    # https://arxiv.org/pdf/2111.04877.pdf
    FEDBUFF = "fedbuff"
    FEDPROX = "fedprox"  # FedProx

    DEFAULT = FEDAVG


class SelectorType(str, Enum):
    """Define selector types."""

    DEFAULT = "default"
    RANDOM = "random"
    FEDBUFF = "fedbuff"


class Job(FlameSchema):
    job_id: str = Field(alias="id")
    name: str


class Registry(FlameSchema):
    sort: RegistryType
    uri: str


class Selector(FlameSchema):
    sort: SelectorType = Field(default=SelectorType.DEFAULT)
    kwargs: dict = Field(default={})


class Optimizer(FlameSchema):
    sort: OptimizerType = Field(default=OptimizerType.DEFAULT)
    kwargs: dict = Field(default={})


class BaseModel(FlameSchema):
    name: str = Field(default="")
    version: int = Field(default=0)


class Hyperparameters(FlameSchema):
    batch_size: t.Optional[int] = Field(alias="batchSize")
    learning_rate: t.Optional[float] = Field(alias="learningRate")
    rounds: int
    epochs: int
    aggregation_goal: t.Optional[int] = Field(alias="aggGoal", default=None)


class Groups(FlameSchema):
    param_channel: str
    global_channel: str


class FuncTags(FlameSchema):
    aggregator: list[str]
    trainer: list[str]


class GroupBy(FlameSchema):
    type: t.Optional[str] = Field(default="")
    value: t.Optional[list[str]] = Field(default=[])

    def groupable_value(self, group_association: str = ""):
        """Return groupby value."""
        if self.value is None:
            return GROUPBY_DEFAULT_GROUP

        if group_association in self.value:
            return group_association

        return GROUPBY_DEFAULT_GROUP

class Broker(FlameSchema):
    sort_to_host: dict


class Channel(FlameSchema):
    name: str
    pair: list[str] = Field(min_length=2)
    is_bidirectional: t.Optional[bool] = Field(default=True)
    group_by: t.Optional[GroupBy] = Field(default=GroupBy())
    func_tags: dict = Field(default={}, alias="func_tags")
    description: t.Optional[str]


class ChannelConfigs(FlameSchema):
    backends: dict = Field(default={})
    channel_brokers: dict = Field(default={})


class Config(FlameSchema):

    def __init__(self, config_path: str):
        raw_config = read_config(config_path)
        transformed_config = transform_config(raw_config)

        super().__init__(**transformed_config)

    role: str
    realm: t.Optional[str]  # to be deprecated
    group_association: dict
    task: t.Optional[str] = Field(default="local")
    task_id: str
    backend: BackendType
    channels: dict
    hyperparameters: Hyperparameters
    brokers: Broker
    job: Job
    registry: Registry
    selector: Selector
    optimizer: t.Optional[Optimizer] = Field(default=Optimizer())
    channel_configs: t.Optional[ChannelConfigs]
    dataset: str
    max_run_time: int
    base_model: BaseModel
    groups: t.Optional[Groups]
    dependencies: list[str]
    func_tag_map: t.Optional[dict]


def read_config(filename: str) -> dict:
    with open(filename) as f:
        return json.loads(f.read())


def transform_config(raw_config: dict) -> dict:
    config_data = {
        "role": raw_config["role"],
        "realm": raw_config["realm"],
        "task_id": raw_config["taskid"],
        "backend": raw_config["backend"],
        "group_association": raw_config["groupAssociation"],
    }

    if raw_config.get("task", None):
        config_data = config_data | {
            "task": raw_config["task"],
        }

    channels, func_tag_map = transform_channels(config_data["role"],
                                                raw_config["channels"])
    config_data = config_data | {
        "channels": channels,
        "func_tag_map": func_tag_map
    }

    hyperparameters = transform_hyperparameters(raw_config["hyperparameters"])
    config_data = config_data | {"hyperparameters": hyperparameters}

    sort_to_host = transform_brokers(raw_config["brokers"])
    config_data = config_data | {"brokers": sort_to_host}

    config_data = config_data | {
        "job": raw_config["job"],
        "registry": raw_config["registry"],
        "selector": raw_config["selector"],
    }

    if raw_config.get("optimizer", None):
        config_data = config_data | {"optimizer": raw_config.get("optimizer")}

    backends, channel_brokers = transform_channel_configs(
        raw_config.get("channelConfigs", {}))
    config_data = config_data | {
        "channel_configs": {
            "backends": backends,
            "channel_brokers": channel_brokers
        }
    }

    config_data = config_data | {
        "dataset": raw_config.get("dataset", ""),
        "max_run_time": raw_config.get("maxRunTime", 300),
        "base_model": raw_config.get("baseModel", None),
        "dependencies": raw_config.get("dependencies", None),
    }

    return config_data


def transform_channel(raw_channel_config: dict):
    name = raw_channel_config["name"]
    pair = raw_channel_config["pair"]
    is_bidirectional = raw_channel_config.get("isBidirectional", True)
    group_by = {
        "type": "",
        "value": []
    } | raw_channel_config.get("groupBy", {})
    func_tags = raw_channel_config.get("funcTags", {})
    description = raw_channel_config.get("description", "")

    return {
        "name": name,
        "pair": pair,
        "is_bidirectional": is_bidirectional,
        "group_by": group_by,
        "func_tags": func_tags,
        "description": description,
    }


def transform_channels(role, raw_channels_config: dict):
    channels = {}
    func_tag_map = {}
    for raw_channel_config in raw_channels_config:
        channel = transform_channel(raw_channel_config)
        channels[channel["name"]] = Channel(**channel)

        for tag in channel["func_tags"][role]:
            func_tag_map[tag] = channel["name"]

    return channels, func_tag_map


def transform_hyperparameters(raw_hyperparameters_config: dict):
    hyperparameters = DEFAULT_HYPERARAMETERS_DICT
    if raw_hyperparameters_config:
        hyperparameters = hyperparameters | raw_hyperparameters_config

    return hyperparameters


def transform_brokers(raw_brokers_config: dict):
    sort_to_host = {}
    for raw_broker in raw_brokers_config:
        sort = raw_broker["sort"]
        host = raw_broker["host"]
        sort_to_host[sort] = host

    return Broker(sort_to_host=sort_to_host)


def transform_channel_configs(raw_channel_configs_config: dict):
    backends = {}
    channel_brokers = {}

    for k, v in raw_channel_configs_config.items():
        backends[k] = v["backend"]
        channel_brokers[k] = transform_brokers(v["brokers"])

    return backends, channel_brokers
