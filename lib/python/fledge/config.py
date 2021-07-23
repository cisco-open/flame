import json
import sys

CONF_KEY_MQTT_BROKER = 'broker'
CONF_KEY_BACKEND = 'backend'
CONF_KEY_AGENT = 'agent'
CONF_KEY_JOB = 'job'
CONF_KEY_JOB_ID = 'id'
CONF_KEY_JOB_NAME = 'name'
CONF_KEY_ROLE = 'role'
CONF_KEY_CHANNEL = 'channel'

CONF_KEY_CHANNEL_NAME = 'name'
CONF_KEY_CHANNEL_PAIR = 'pair'
CONF_KEY_CHANNEL_IS_BIDIR = 'isBidirectional'
CONF_KEY_CHANNEL_GROUPBY = 'groupBy'
CONF_KEY_CHANNEL_GROUPBY_TYPE = 'type'
CONF_KEY_CHANNEL_GROUPBY_VALUE = 'value'

BACKEND_TYPE_LOCAL = 'local'
BACKEND_TYPE_P2P = 'p2p'
BACKEND_TYPE_MQTT = 'mqtt'

backend_types = [BACKEND_TYPE_LOCAL, BACKEND_TYPE_P2P, BACKEND_TYPE_MQTT]


class Config(object):
    class Job(object):
        def __init__(self, json_data):
            self.job_id = json_data[CONF_KEY_JOB_ID]
            self.name = json_data[CONF_KEY_JOB_NAME]

        def print(self):
            print('\t--- job ---')
            print(f'\t{CONF_KEY_JOB_ID}: {self.job_id}')
            print(f'\t{CONF_KEY_JOB_NAME}: {self.name}')

    class Channel(object):
        class GroupBy(object):
            def __init__(self, json_data):
                self.gtype = json_data[CONF_KEY_CHANNEL_GROUPBY_TYPE]
                self.value = json_data[CONF_KEY_CHANNEL_GROUPBY_VALUE]

            def print(self):
                print('\t\t--- groupby ---')
                print(f'\t\t{CONF_KEY_CHANNEL_GROUPBY_TYPE}: {self.gtype}')
                print(f'\t\t{CONF_KEY_CHANNEL_GROUPBY_VALUE}: {self.value}')

        def __init__(self, json_data):
            self.name = json_data[CONF_KEY_CHANNEL_NAME]
            self.pair = json_data[CONF_KEY_CHANNEL_PAIR]
            if len(self.pair) != 2:
                sys.exit(f'A pair must have two ends, but got {self.pair}')

            self.is_bidrectional = True
            if CONF_KEY_CHANNEL_IS_BIDIR in json_data:
                self.is_bidrectional = json_data[CONF_KEY_CHANNEL_IS_BIDIR]

            self.groupby = None
            if CONF_KEY_CHANNEL_GROUPBY in json_data:
                self.groupby = Config.Channel.GroupBy(
                    json_data[CONF_KEY_CHANNEL_GROUPBY]
                )

        def print(self):
            print('\t--- channel ---')
            print(f'\t{CONF_KEY_CHANNEL_NAME}: {self.name}')
            print(f'\t{CONF_KEY_CHANNEL_PAIR}: {self.pair}')
            print(f'\t{CONF_KEY_CHANNEL_IS_BIDIR}: {self.is_bidrectional}')
            self.groupby.print()

    def __init__(self, config_file: str):
        with open(config_file) as f:
            json_data = json.load(f)
            f.close()

        self.broker = json_data[CONF_KEY_MQTT_BROKER]
        self.backend = json_data[CONF_KEY_BACKEND]
        if not self.is_valid(self.backend, backend_types):
            sys.exit(f'not a vailid backend type: {self.backend}')

        self.agent = json_data[CONF_KEY_AGENT]
        self.job = Config.Job(json_data[CONF_KEY_JOB])
        self.role = json_data[CONF_KEY_ROLE]
        self.channels = {}
        for channel_info in json_data[CONF_KEY_CHANNEL]:
            channel_config = Config.Channel(channel_info)
            self.channels[channel_config.name] = channel_config

    def is_valid(self, needle, haystack):
        return needle in haystack

    def print(self):
        print('--- config ---')
        print(f'{CONF_KEY_MQTT_BROKER}: {self.broker}')
        print(f'{CONF_KEY_BACKEND}: {self.backend}')
        print(f'{CONF_KEY_AGENT}: {self.agent}')
        print(f'{CONF_KEY_ROLE}: {self.role}')
        self.job.print()
        for _, channel in self.channels.items():
            channel.print()
