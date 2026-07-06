# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License"); you
# may not use this file except in compliance with the License. You may
# obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""Config parser."""

import json
import typing as t
from enum import Enum

from pydantic import BaseModel as pydBaseModel
from pydantic import ConfigDict, Extra, Field


class FlameSchema(pydBaseModel):
    model_config = ConfigDict(populate_by_name=True)


GROUPBY_DEFAULT_GROUP = "default"
GROUP_ASSOCIATION_SEPARATOR = "/"
DEFAULT_HYPERARAMETERS_DICT = {"rounds": 1, "epochs": 1, "batchSize": 16}

RAW_KEY_BACKEND = "backend"
RAW_KEY_BROKER_HOST = "brokerHost"


class BackendType(str, Enum):
    """Define backend types."""

    LOCAL = "local"
    P2P = "p2p"
    MQTT = "mqtt"
    SHM = "shm"


class RegistryType(str, Enum):
    """Define model registry types."""

    DUMMY = "dummy"
    LOCAL = "local"
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
    FEDDYN = "feddyn"
    SCAFFOLD = "scaffold"
    FEDGFT = "fedgft"
    REFL = "refl"  # REFL: Resource-Efficient Federated Learning with staleness-aware aggregation

    DEFAULT = FEDAVG


class SelectorType(str, Enum):
    """Define selector types."""

    DEFAULT = "default"
    RANDOM = "random"
    FEDBUFF = "fedbuff"
    OORT = "oort"
    ASYNC_OORT = "async_oort"
    REFL_OORT = "refl_oort"  # REFL-enhanced Oort with priority selection and availability tracking
    ASYNC_RANDOM = "async_random"
    FEDDANCE = "feddance"  # FedDance: Poisson V_m, loss I_m, accuracy slope A_m, UCB exploration
    ORACLE = "oracle"  # Streaming-misprioritization ceiling: greedy top-K by fresh true utility


class DataSamplerType(str, Enum):
    """Define datasampler types."""

    DEFAULT = "default"
    FEDBALANCER = "fedbalancer"


class PrivacyType(str, Enum):
    """Define privacy policy types."""

    DEFAULT = "default"
    DP = "dp"


class TrainerAvailState(str, Enum):
    """Define availability status types."""

    AVL_TRAIN = "AVL_TRAIN"
    AVL_EVAL = "AVL_EVAL"
    UN_AVL = "UN_AVL"


class Job(FlameSchema):
    job_id: str = Field(alias="id")
    name: str


class Registry(FlameSchema):
    sort: RegistryType
    uri: str


class Selector(FlameSchema):
    sort: SelectorType = Field(default=SelectorType.DEFAULT)
    kwargs: dict = Field(default={})


class DataSampler(FlameSchema):
    sort: DataSamplerType = Field(default=DataSamplerType.DEFAULT)
    kwargs: dict = Field(default={})


class Privacy(FlameSchema):
    sort: PrivacyType = Field(default=PrivacyType.DEFAULT)
    kwargs: dict = Field(default={})


class Optimizer(FlameSchema):
    sort: OptimizerType = Field(default=OptimizerType.DEFAULT)
    kwargs: dict = Field(default={})


class BaseModel(FlameSchema):
    name: str = Field(default="")
    version: int = Field(default=0)


class Hyperparameters(FlameSchema, extra=Extra.allow):
    batch_size: t.Optional[int] = Field(alias="batchSize", default=None)
    learning_rate: t.Optional[float] = Field(alias="learningRate", default=None)
    weight_decay: t.Optional[float] = Field(alias="weightDecay", default=None)
    rounds: int
    epochs: int
    aggregation_goal: t.Optional[int] = Field(alias="aggGoal", default=None)
    eval_every_n_rounds: t.Optional[int] = Field(alias="evalEveryNRounds", default=50)
    eval_goal_factor: t.Optional[float] = Field(alias="evalGoalFactor", default=None)
    # Target-accuracy stopping: stop once test accuracy stays >= target for
    # `stable_evals_above_target` consecutive evals (resets on any dip). The
    # existing `rounds` / `max_experiment_runtime_s` caps remain as the safety net so a
    # non-converging run still terminates. None disables the rule.
    target_accuracy: t.Optional[float] = Field(alias="targetAccuracy", default=None)
    stable_evals_above_target: t.Optional[int] = Field(
        alias="stableEvalsAboveTarget", default=20
    )
    round_nudge_type: t.Optional[str] = Field(
        alias="roundNudgeType", default="last_train"
    )
    # Deterministic RNG seed: seeds the global RNGs (torch model init, syncfl
    # internal_init) and each selector's dedicated RNG, making selection
    # reproducible across real/sim. None = unseeded.
    seed: t.Optional[int] = Field(alias="seed", default=None)
    # TODO: concurrency is for coordinator in coordinated asyncfl this
    #       is a workaround since there is no per-role config
    #       mechanism in the control plane. This needs to be revisited
    #       (perhaps removed) once per-role config functionality is in
    #       place in the control plane.
    concurrency: t.Optional[int] = Field(alias="concurrency", default=None)
    track_trainer_avail: t.Optional[dict] = Field(alias="trackTrainerAvail", default={})
    reject_stale_updates: t.Optional[bool] = Field(
        alias="rejectStaleUpdates", default=False
    )
    # FedFwd (fwdllm/fwdllm_plus/fluxtune) staleness gate on incoming trainer
    # updates, checked in fwdllm_aggregator._process_single_trainer_message:
    #   "exact"         -- reject unless the update matches the aggregator's
    #                      current (round, data_id, iteration_per_data_id)
    #                      exactly. Strict-sync baseline (fwdllm).
    #   "round_data_id" -- reject unless (round, data_id) match; any
    #                      iteration_per_data_id within that data_id is
    #                      accepted. fwdllm_plus.
    #   "none"          -- no staleness gate (any version accepted); FedFwd's
    #                      async baseline (fluxtune) relies on stale/
    #                      in-flight updates by design.
    # None (unset) falls back to reject_stale_updates above, for examples
    # that only know that older boolean knob.
    staleness_policy: t.Optional[str] = Field(alias="stalenessPolicy", default=None)
    heartbeats: t.Optional[dict] = Field(alias="heartbeats", default={})
    client_notify: t.Optional[dict] = Field(
        alias="clientAvailAwareNotify", default=None
    )
    training_delay_enabled: t.Optional[bool] = Field(
        alias="trainingDelayEnabled", default=False
    )
    training_delay_s: t.Optional[float] = Field(
        alias="trainingDelaySeconds", default=None
    )
    training_delay_factor: t.Optional[float] = Field(
        alias="trainingDelayFactor", default=None
    )
    # Sim-mode per-commit virtual-clock overhead (MQTT/dispatch). 0 = off.
    sim_commit_overhead_s: t.Optional[float] = Field(
        alias="simCommitOverheadSeconds", default=0.0
    )
    # Sim PRE-commit holding leg added to trainer sct (counts toward staleness).
    sim_completion_leg_s: t.Optional[float] = Field(
        alias="simCompletionLegSeconds", default=0.0
    )
    # Sim POST-commit re-dispatch cooldown; spaces completions without inflating staleness.
    sim_redispatch_gap_s: t.Optional[float] = Field(
        alias="simRedispatchGapSeconds", default=0.0
    )
    # Sim async-stack: cap each commit's clock advance at the earliest in-flight
    # FUTURE modeled completion (+slack), so a forced far-future straggler commit
    # can't lap the fresh fast cohort still mid-flight (the dominant past-dating
    # source). Re-bases the inert arrival-gate onto modeled completion.
    sim_clock_jump_clamp: t.Optional[bool] = Field(
        alias="simClockJumpClamp", default=True
    )
    # Sim async-stack: event-driven re-dispatch. Stamp each TRAIN dispatch at the
    # vclock its slot freed (a prior commit) instead of one frozen round-start
    # frontier, so the per-trainer completion stagger (sct = sim_send_ts + compute)
    # is preserved across the round boundary as it is in real. Sync stays batched.
    sim_staggered_redispatch: t.Optional[bool] = Field(
        alias="simStaggeredRedispatch", default=False
    )
    # Sim async-stack: ingest in-flight updates into the sct-ordered reorder
    # buffer by draining each end's rx queue DIRECTLY (channel.drain_ready),
    # instead of through the recv_fifo streamer. The streamer's background task +
    # shared queue can strand a delivered update where the readiness probe can't
    # see it, so the buffer commits an incomplete subset and the virtual clock
    # laps the stranded (lower-sct) updates → they commit past-dated (async
    # staleness ~15 vs real ~3; only ~1.6 of 10 commits/round advance the clock).
    # Draining directly keeps the buffer a COMPLETE snapshot of arrived in-flight
    # updates so the existing min-sct gate commits in true completion order.
    # Default off ⇒ recv_fifo path (byte-identical to today). Sync untouched.
    sim_sct_ordered_drain: t.Optional[bool] = Field(
        alias="simSctOrderedDrain", default=False
    )
    # Real-only settle sleep before selection (hit 2x/commit). 0 = compute-bound.
    real_distribute_settle_s: t.Optional[float] = Field(
        alias="realDistributeSettleSeconds", default=0.1
    )
    # Hold a dispatched trainer in-flight (occupying its concurrency slot, out of the
    # eligible pool) until its update commits, instead of freeing the slot at instant
    # physical arrival — so the committed/eligible mix matches real. Sync stack (oort):
    # adds the still-computing set to the unavailable list (§4.5). Async stack (felix):
    # widens _sim_hold_busy_slots to the full dispatched-but-not-committed set, held via
    # selected_ends (a slot), NOT the unavailable list. Default off ⇒ holds only the
    # already-buffered set.
    sim_inflight_residence: t.Optional[bool] = Field(
        alias="simInflightResidence", default=False
    )
    # Sim sync-stack: keep a prior-round straggler still computing at round start
    # (sct > vclock_round_start) buffered and carried in-flight until vclock >= sct,
    # instead of popping + stale-rejecting it on instant arrival (which drains sim's
    # in-flight to ~0 while real carries the overcommit). Gates carry/cleanup, whereas
    # sim_inflight_residence gates pool re-entry.
    sim_inflight_carryover: t.Optional[bool] = Field(
        alias="simInflightCarryover", default=False
    )
    use_oort_loss_fn: t.Optional[str] = Field(alias="useOORTLossFn", default="False")
    wait_until_next_avl: t.Optional[bool] = Field(
        alias="waitUntilNextAvail", default=False
    )
    # Sim unavailability feature gate (§1 / §8.4). Default False ⇒ byte-identical
    # to all existing runs. Set True to activate the oracular trace-read path.
    sim_unavailability: t.Optional[bool] = Field(
        alias="simUnavailability", default=False
    )
    # Per-baseline: aware baselines free stalled slots proactively at the next
    # selection boundary (Stage D); unaware wait for the 90s vclock abandon.
    # Kept for backward compat — new code reads proactive_inflight_evict first.
    availability_aware: t.Optional[bool] = Field(
        alias="availabilityAware", default=False
    )
    # Two-axis flag split (T1): avail_select_filter gates selection filtering;
    # proactive_inflight_evict gates felix-only boundary eviction.
    avail_select_filter: t.Optional[bool] = Field(
        alias="availSelectFilter", default=True
    )
    proactive_inflight_evict: t.Optional[bool] = Field(
        alias="proactiveInflightEvict", default=None
    )
    # Override directory for availability trace YAMLs. Defaults to
    # examples/_metadata/availability_traces/ when None.
    availability_trace_dir: t.Optional[str] = Field(
        alias="availabilityTraceDir", default=None
    )
    inc_model_version_per_data_id: t.Optional[bool] = Field(
        alias="incModelVersionPerDataId", default=False
    )
    # Expected keys: enabled (bool), period_s (float), amplitude_fraction (float)
    # Defaults: enabled=False, period_s=120, amplitude_fraction=0.2
    training_delay_variation: t.Optional[dict] = None

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
    description: t.Optional[str] = None
    backend: t.Optional[str] = None
    broker_host: t.Optional[str] = None


class ChannelConfigs(FlameSchema):
    backends: dict = Field(default={})
    channel_brokers: dict = Field(default={})


class Config(FlameSchema):
    def __init__(self, config_path: str):
        raw_config = read_config(config_path)
        transformed_config = transform_config(raw_config)

        super().__init__(**transformed_config)

    role: str
    realm: t.Optional[str] = None  # to be deprecated
    group_association: dict
    task: t.Optional[str] = Field(default="local")
    task_id: str
    backend: BackendType
    channels: dict
    hyperparameters: t.Optional[Hyperparameters] = None
    brokers: Broker
    job: Job
    registry: t.Optional[Registry] = None
    selector: t.Optional[Selector] = None
    datasampler: t.Optional[DataSampler] = Field(default=DataSampler())
    privacy: t.Optional[Privacy] = Field(default=Privacy())
    optimizer: t.Optional[Optimizer] = Field(default=Optimizer())
    dataset: str
    max_run_time: int
    base_model: t.Optional[BaseModel] = None
    groups: t.Optional[Groups] = None
    dependencies: t.Optional[list[str]] = None
    func_tag_map: t.Optional[dict] = None


def read_config(filename: str) -> dict:
    with open(filename) as f:
        return json.loads(f.read())


def transform_config(raw_config: dict) -> dict:
    config_data = {
        "role": raw_config["role"],
        "realm": raw_config["realm"],
        "task_id": raw_config["taskid"],
        "backend": raw_config[RAW_KEY_BACKEND],
        "group_association": raw_config["groupAssociation"],
    }

    if raw_config.get("task", None):
        config_data = config_data | {
            "task": raw_config["task"],
        }

    channels, func_tag_map = transform_channels(
        config_data["role"], raw_config["channels"]
    )
    config_data = config_data | {"channels": channels, "func_tag_map": func_tag_map}

    if raw_config.get("hyperparameters", None):
        hyperparameters = transform_hyperparameters(raw_config["hyperparameters"])

        config_data = config_data | {"hyperparameters": hyperparameters}

    sort_to_host = transform_brokers(raw_config["brokers"])
    config_data = config_data | {"brokers": sort_to_host}

    config_data = config_data | {
        "job": raw_config["job"],
        "selector": raw_config["selector"],
    }

    if raw_config.get("registry", None):
        config_data = config_data | {"registry": raw_config["registry"]}

    if raw_config.get("optimizer", None):
        config_data = config_data | {"optimizer": raw_config.get("optimizer")}

    if raw_config.get("datasampler", None):
        raw_config["datasampler"]["kwargs"].update(hyperparameters)
        config_data = config_data | {"datasampler": raw_config.get("datasampler")}

    if raw_config.get("privacy", None):
        config_data = config_data | {"privacy": raw_config.get("privacy")}

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
    group_by = {"type": "", "value": []} | raw_channel_config.get("groupBy", {})
    func_tags = raw_channel_config.get("funcTags", {})
    description = raw_channel_config.get("description", "")

    backend = raw_channel_config.get(RAW_KEY_BACKEND, "")
    broker_host = raw_channel_config.get(RAW_KEY_BROKER_HOST, "")

    return {
        "name": name,
        "pair": pair,
        "is_bidirectional": is_bidirectional,
        "group_by": group_by,
        "func_tags": func_tags,
        "description": description,
        "backend": backend,
        "broker_host": broker_host,
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
