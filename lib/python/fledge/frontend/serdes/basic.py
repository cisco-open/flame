import json

from .abstract import AbstractSerDes


class BasicSerDes(AbstractSerDes):
    def __init__(self):
        self._encoding = 'utf-8'

    def to_bytes(self, obj_data):
        json_data = self.to_json(obj_data)
        return bytes(json_data, self._encoding)

    def to_json(self, obj_data):
        return json.dumps(obj_data)

    def to_object(self, byte_data):
        json_data = str(byte_data, self._encoding)
        return json.loads(json_data)
