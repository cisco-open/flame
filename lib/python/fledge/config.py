# Copyright (c) 2021 Cisco Systems, Inc. and its affiliates
# All rights reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Config parser."""

import json
import sys
from enum import Enum

CONF_KEY_AGENT = 'agent'
CONF_KEY_AGENT_ID = 'agentid'
CONF_KEY_BACKEND = 'backend'
CONF_KEY_CHANNEL = 'channels'
CONF_KEY_DATASET = 'dataset'
CONF_KEY_HYPERPARAMS = 'hyperparameters'
CONF_KEY_MAX_RUN_TIME = 'maxRunTime'
CONF_KEY_REALM = 'realm'
CONF_KEY_ROLE = 'role'

CONF_KEY_CHANNEL_NAME = 'name'
CONF_KEY_CHANNEL_PAIR = 'pair'
CONF_KEY_CHANNEL_IS_BIDIR = 'isBidirectional'
CONF_KEY_CHANNEL_GROUPBY = 'groupBy'
CONF_KEY_CHANNEL_GROUPBY_TYPE = 'type'
CONF_KEY_CHANNEL_GROUPBY_VALUE = 'value'
CONF_KEY_CHANNEL_FUNC_TAGS = 'funcTags'

CONF_KEY_BASE_MODEL = 'baseModel'
CONF_KEY_BASE_MODEL_NAME = 'name'
CONF_KEY_BASE_MODEL_VERSION = 'version'

CONF_KEY_BROKERS = 'brokers'
CONF_KEY_BROKERS_HOST = 'host'
CONF_KEY_BROKERS_SORT = 'sort'

CONF_KEY_JOB = 'job'
CONF_KEY_JOB_ID = 'id'
CONF_KEY_JOB_NAME = 'name'

CONF_KEY_REGISTRY = 'registry'
CONF_KEY_REGISTRY_SORT = 'sort'
CONF_KEY_REGISTRY_URI = 'uri'

GROUPBY_DEFAULT_GROUP = 'default'


class BackendType(Enum):
    """Define backend types."""

    LOCAL = 1
    P2P = 2
    MQTT = 3


class RegistryType(Enum):
    """Define model registry types."""

    MLFLOW = 1


REALM_SEPARATOR = '/'


class Config(object):
    """Config class."""

    #
    class Channel(object):
        """Channel class."""

        #
        class GroupBy(object):
            """GroupBy class."""

            #
            def __init__(self, json_data=None):
                """Initialize GroupBy instance."""
                self.gtype = ''
                self.value = []

                if json_data is None:
                    return

                self.gtype = json_data[CONF_KEY_CHANNEL_GROUPBY_TYPE]
                self.value = json_data[CONF_KEY_CHANNEL_GROUPBY_VALUE]

            def __str__(self):
                """Return GroupBy info as string."""
                return (
                    "\t\t--- groupby ---\n" +
                    f"\t\t\t{CONF_KEY_CHANNEL_GROUPBY_TYPE}: {self.gtype}\n" +
                    f"\t\t\t{CONF_KEY_CHANNEL_GROUPBY_VALUE}: {self.value}\n")

            def groupable_value(self, realm=''):
                """Return groupby value."""
                if self.value is None:
                    return GROUPBY_DEFAULT_GROUP

                for entry in self.value:
                    # check if an entry is a prefix of realm in a '/'-separated
                    # fashion; if so, then return the matching entry
                    if realm.startswith(entry) and (len(realm) == len(entry)
                                                    or realm[len(entry)]
                                                    == REALM_SEPARATOR):
                        return entry

                return GROUPBY_DEFAULT_GROUP

        def __init__(self, json_data):
            """Initialize Channel instance."""
            self.name = json_data[CONF_KEY_CHANNEL_NAME]
            self.pair = json_data[CONF_KEY_CHANNEL_PAIR]
            if len(self.pair) != 2:
                sys.exit(f'A pair must have two ends, but got {self.pair}')

            self.is_bidrectional = True
            if CONF_KEY_CHANNEL_IS_BIDIR in json_data:
                self.is_bidrectional = json_data[CONF_KEY_CHANNEL_IS_BIDIR]

            if CONF_KEY_CHANNEL_GROUPBY in json_data:
                self.groupby = Config.Channel.GroupBy(
                    json_data[CONF_KEY_CHANNEL_GROUPBY])
            else:
                self.groupby = Config.Channel.GroupBy()

            self.func_tags = dict()
            if CONF_KEY_CHANNEL_FUNC_TAGS in json_data:
                for k, v in json_data[CONF_KEY_CHANNEL_FUNC_TAGS].items():
                    # k: role, v: tag list
                    self.func_tags[k] = v

        def __str__(self):
            """Return channel info as string."""
            return (
                "\t--- channel ---\n" +
                f"\t\t{CONF_KEY_CHANNEL_NAME}: {self.name}\n" +
                f"\t\t{CONF_KEY_CHANNEL_PAIR}: {self.pair}\n" +
                f"\t\t{CONF_KEY_CHANNEL_IS_BIDIR}: {self.is_bidrectional}\n" +
                str(self.groupby))

    class BaseModel(object):
        """Base model class."""

        #
        def __init__(self, json_data=None):
            """Initialize BaseModel instance."""
            self.name = json_data[CONF_KEY_BASE_MODEL_NAME]
            self.version = json_data[CONF_KEY_BASE_MODEL_VERSION]

        def __str__(self):
            """Return base model's detail as string."""
            return ("\t--- base model ---\n" +
                    f"\t\t{CONF_KEY_BASE_MODEL_NAME}: {self.name}\n" +
                    f"\t\t{CONF_KEY_BASE_MODEL_VERSION}: {self.version}\n")

    class Brokers(object):
        """Brokers class."""

        #
        def __init__(self, json_data=None):
            """Initialize BaseModel instance."""
            self.sort_to_host = dict()

            for broker in json_data:
                key = broker[CONF_KEY_BROKERS_SORT].upper()
                try:
                    sort = BackendType[key]
                except KeyError:
                    valid_types = [backend.name for backend in BackendType]
                    sys.exit(f"invalid sort type: {key}\n" +
                             f"broker's sort must be one of {valid_types}.")

                host = broker[CONF_KEY_BROKERS_HOST]
                self.sort_to_host[sort] = host

        def __str__(self):
            """Return brokers' details as string."""
            info = ""
            for sort, host in self.sort_to_host.items():
                info += f"\t\t{sort}: {host}\n"
            return ("\t--- brokers ---\n" + info)

    class Job(object):
        """Job class."""

        #
        def __init__(self, json_data=None):
            """Initialize Job instance."""
            self.job_id = json_data[CONF_KEY_JOB_ID]
            self.name = json_data[CONF_KEY_JOB_NAME]

        def __str__(self):
            """Return job's detail in string format."""
            return ("\t--- job ---\n" +
                    f"\t\t{CONF_KEY_JOB_ID}: {self.job_id}\n" +
                    f"\t\t{CONF_KEY_JOB_NAME}: {self.name}\n")

    class Registry(object):
        """Registry class."""

        #
        def __init__(self, json_data=None):
            """Initialize Registry instance."""
            sort = json_data[CONF_KEY_REGISTRY_SORT].upper()
            try:
                self.sort = RegistryType[sort]
            except KeyError:
                valid_types = [registry.name for registry in RegistryType]
                sys.exit(f"invailid registry type: {sort}" +
                         f"valid registry type(s) are {valid_types}")

            self.uri = json_data[CONF_KEY_REGISTRY_URI]

        def __str__(self):
            """Return model registry's detail in string format."""
            return ("\t--- registry ---\n" +
                    f"\t\t{CONF_KEY_REGISTRY_SORT}: {self.sort}\n" +
                    f"\t\t{CONF_KEY_REGISTRY_URI}: {self.uri}\n")

    def __init__(self, config_file: str):
        """Initialize Config instance."""
        with open(config_file) as f:
            json_data = json.load(f)
            f.close()

        self.agent = 'local'
        if CONF_KEY_AGENT in json_data:
            self.agent = json_data[CONF_KEY_AGENT]

        self.agent_id = json_data[CONF_KEY_AGENT_ID]

        backend_key = json_data[CONF_KEY_BACKEND].upper()
        try:
            self.backend = BackendType[backend_key]
        except KeyError:
            valid_types = [backend.name for backend in BackendType]
            sys.exit(f"invailid backend type: {backend_key}\n" +
                     f"valid backend type(s) are {valid_types}")

        self.brokers = Config.Brokers(json_data[CONF_KEY_BROKERS])

        self.dataset = ''
        if CONF_KEY_DATASET in json_data:
            self.dataset = json_data[CONF_KEY_DATASET]

        self.hyperparameters = None
        if CONF_KEY_HYPERPARAMS in json_data:
            self.hyperparameters = json_data[CONF_KEY_HYPERPARAMS]

        self.max_run_time = 300
        if CONF_KEY_MAX_RUN_TIME in json_data:
            self.max_run_time = json_data[CONF_KEY_MAX_RUN_TIME]

        self.role = json_data[CONF_KEY_ROLE]
        self.realm = json_data[CONF_KEY_REALM]

        self.func_tag_map = dict()
        self.channels = dict()

        for channel_info in json_data[CONF_KEY_CHANNEL]:
            channel_config = Config.Channel(channel_info)
            self.channels[channel_config.name] = channel_config

            # build a map from function tag to channel name
            for tag in channel_config.func_tags[self.role]:
                self.func_tag_map[tag] = channel_config.name

        self.base_model = None
        if CONF_KEY_BASE_MODEL in json_data:
            self.base_model = Config.BaseModel(json_data[CONF_KEY_BASE_MODEL])

        self.job = Config.Job(json_data[CONF_KEY_JOB])
        self.registry = Config.Registry(json_data[CONF_KEY_REGISTRY])

    def __str__(self):
        """Return config info as string."""
        info = ("--- config ---\n" +
                f"\t{CONF_KEY_BACKEND}: {self.backend}\n" +
                f"\t{CONF_KEY_AGENT}: {self.agent}\n" +
                f"\t{CONF_KEY_ROLE}: {self.role}\n" +
                f"\t{CONF_KEY_REALM}: {self.realm}\n" + str(self.base_model) +
                str(self.brokers) + str(self.job) + str(self.registry))
        for _, channel in self.channels.items():
            info += str(channel)

        return info
