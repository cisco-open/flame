import json
from collections import OrderedDict

import numpy as np
import torch

from .basic import BasicMessage


class PyTorchMessage(BasicMessage):
    def __init__(self, data=None):
        super().__init__(data)

    def to_json(self):
        orig_state = None

        if self.STATE in self.data:
            orig_state = self.data[self.STATE]

            list_dict = OrderedDict()

            for key, tensor in self.data[self.STATE].items():
                list_dict[key] = tensor.tolist()
            self.data[self.STATE] = list_dict

        json_data = json.dumps(self.data)

        # restore state in original format
        if orig_state:
            self.data[self.STATE] = orig_state

        return json_data

    @classmethod
    def to_object(cls, byte_data):
        if not byte_data:
            return None

        json_data = str(byte_data, cls._encoding)

        try:
            data = json.loads(json_data)
        except json.decoder.JSONDecodeError:
            return None

        if cls.STATE in data:
            list_dict = OrderedDict()
            for key, val in data[cls.STATE].items():
                list_dict[key] = torch.from_numpy(np.asarray(val))

            data[cls.STATE] = list_dict

        return cls(data)
