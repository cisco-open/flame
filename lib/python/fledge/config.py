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

CONF_KEY_AGENT = 'agent'
CONF_KEY_AGENT_ID = 'agentid'
CONF_KEY_BACKEND = 'backend'
CONF_KEY_CHANNEL = 'channels'
CONF_KEY_DATASET = 'dataset'
CONF_KEY_HYPERPARAMS = 'hyperparameters'
CONF_KEY_JOB_ID = 'jobid'
CONF_KEY_MAX_RUN_TIME = 'maxRunTime'
CONF_KEY_MQTT_BROKER = 'broker'
CONF_KEY_REALM = 'realm'
CONF_KEY_ROLE = 'role'

CONF_KEY_CHANNEL_NAME = 'name'
CONF_KEY_CHANNEL_PAIR = 'pair'
CONF_KEY_CHANNEL_IS_BIDIR = 'isBidirectional'
CONF_KEY_CHANNEL_GROUPBY = 'groupBy'
CONF_KEY_CHANNEL_GROUPBY_TYPE = 'type'
CONF_KEY_CHANNEL_GROUPBY_VALUE = 'value'
CONF_KEY_CHANNEL_FUNC_TAGS = 'funcTags'

GROUPBY_DEFAULT_GROUP = 'default'

BACKEND_TYPE_LOCAL = 'local'
BACKEND_TYPE_P2P = 'p2p'
BACKEND_TYPE_MQTT = 'mqtt'

REALM_SEPARATOR = '|'

backend_types = [BACKEND_TYPE_LOCAL, BACKEND_TYPE_P2P, BACKEND_TYPE_MQTT]


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
                """Initialize."""
                self.gtype = ''
                self.value = []

                if json_data is None:
                    return

                self.gtype = json_data[CONF_KEY_CHANNEL_GROUPBY_TYPE]
                self.value = json_data[CONF_KEY_CHANNEL_GROUPBY_VALUE]

            def groupable_value(self, realm=''):
                """Return groupby value."""
                if self.value is None:
                    return GROUPBY_DEFAULT_GROUP

                for entry in self.value:
                    # check if an entry is a prefix of realm in a dot-separated
                    # fashion; if so, then return the matching entry
                    if realm.startswith(entry) and (
                        len(realm) == len(entry) or
                        realm[len(entry)] == REALM_SEPARATOR
                    ):
                        return entry

                return GROUPBY_DEFAULT_GROUP

            def print(self):
                """Print GroupBy info."""
                print('\t\t--- groupby ---')
                print(f'\t\t{CONF_KEY_CHANNEL_GROUPBY_TYPE}: {self.gtype}')
                print(f'\t\t{CONF_KEY_CHANNEL_GROUPBY_VALUE}: {self.value}')

        def __init__(self, json_data):
            """Initialize."""
            self.name = json_data[CONF_KEY_CHANNEL_NAME]
            self.pair = json_data[CONF_KEY_CHANNEL_PAIR]
            if len(self.pair) != 2:
                sys.exit(f'A pair must have two ends, but got {self.pair}')

            self.is_bidrectional = True
            if CONF_KEY_CHANNEL_IS_BIDIR in json_data:
                self.is_bidrectional = json_data[CONF_KEY_CHANNEL_IS_BIDIR]

            if CONF_KEY_CHANNEL_GROUPBY in json_data:
                self.groupby = Config.Channel.GroupBy(
                    json_data[CONF_KEY_CHANNEL_GROUPBY]
                )
            else:
                self.groupby = Config.Channel.GroupBy()

            self.func_tags = list()
            if CONF_KEY_CHANNEL_FUNC_TAGS in json_data:
                self.func_tags = json_data[CONF_KEY_CHANNEL_FUNC_TAGS]

        def print(self):
            """Print channel info."""
            print('\t--- channel ---')
            print(f'\t{CONF_KEY_CHANNEL_NAME}: {self.name}')
            print(f'\t{CONF_KEY_CHANNEL_PAIR}: {self.pair}')
            print(f'\t{CONF_KEY_CHANNEL_IS_BIDIR}: {self.is_bidrectional}')
            self.groupby.print()

    def __init__(self, config_file: str):
        """Initialize."""
        with open(config_file) as f:
            json_data = json.load(f)
            f.close()

        self.agent = 'local'
        if CONF_KEY_AGENT in json_data:
            self.agent = json_data[CONF_KEY_AGENT]

        self.agent_id = json_data[CONF_KEY_AGENT_ID]

        self.backend = json_data[CONF_KEY_BACKEND]
        if not self.is_valid(self.backend, backend_types):
            sys.exit(f'not a vailid backend type: {self.backend}')
        self.broker = json_data[CONF_KEY_MQTT_BROKER]

        self.dataset = ''
        if CONF_KEY_DATASET in json_data:
            self.dataset = json_data[CONF_KEY_DATASET]

        self.hyperparameters = None
        if CONF_KEY_HYPERPARAMS in json_data:
            self.hyperparameters = json_data[CONF_KEY_HYPERPARAMS]

        self.max_run_time = 300
        if CONF_KEY_MAX_RUN_TIME in json_data:
            self.max_run_time = json_data[CONF_KEY_MAX_RUN_TIME]

        self.job_id = json_data[CONF_KEY_JOB_ID]
        self.role = json_data[CONF_KEY_ROLE]
        self.realm = json_data[CONF_KEY_REALM]

        self.func_tag_map = dict()
        self.channels = dict()

        for channel_info in json_data[CONF_KEY_CHANNEL]:
            channel_config = Config.Channel(channel_info)
            self.channels[channel_config.name] = channel_config

            # build a map from function tag to channel name
            for tag in channel_config.func_tags:
                self.func_tag_map[tag] = channel_config.name

    def is_valid(self, needle, haystack):
        """Return if key is found in json data."""
        return needle in haystack

    def print(self):
        """Print config info."""
        print('--- config ---')
        print(f'{CONF_KEY_MQTT_BROKER}: {self.broker}')
        print(f'{CONF_KEY_BACKEND}: {self.backend}')
        print(f'{CONF_KEY_AGENT}: {self.agent}')
        print(f'{CONF_KEY_ROLE}: {self.role}')
        print(f'{CONF_KEY_REALM}: {self.realm}')
        print(f'{CONF_KEY_JOB_ID}: {self.jobid}')
        for _, channel in self.channels.items():
            channel.print()
