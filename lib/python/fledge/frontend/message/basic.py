import json


class BasicMessage(object):
    STATE = 'state'
    _encoding = 'utf-8'

    def __init__(self, data=None):
        self.data = {} if not data else data

    def set_state(self, state):
        '''
        For Keras, state should be weights represented as a list of numpy's
        ndarrays. For PyTorch, state must be state dictionary
        '''
        self.set_attr(self.STATE, state)

    def get_state(self):
        return self.get_attr(self.STATE)

    def reset_state(self):
        self.reset_attr(self.STATE)

    def set_attr(self, k, v):
        '''
        set_attr() is for setting attributes other than state;
        the argument v should be JSON serializable,
        and state should be set by set_state()
        '''
        self.data[k] = v

    def get_attr(self, k):
        if k not in self.data:
            return None

        return self.data[k]

    def reset_attr(self, k):
        if k in self.data:
            del self.data[k]

    def iterate(self):
        return list(self.data.items())

    def to_bytes(self):
        json_data = self.to_json()
        return bytes(json_data, self._encoding)

    def to_json(self):
        return json.dumps(self.data)

    @classmethod
    def to_object(cls, byte_data):
        if not byte_data:
            return None

        json_data = str(byte_data, cls._encoding)

        try:
            data = json.loads(json_data)
        except json.decoder.JSONDecodeError:
            return None

        return cls(data)
