from .abstract import AbstractBackend


class MqttBackend(AbstractBackend):
    async def _setup_server(self):
        pass

    def uid(self):
        pass

    def endpoint(self):
        pass

    def connect(self, end_id, endpoint):
        pass

    def close(self):
        pass

    def send(self, end_id, msg):
        pass

    def recv(self, end_id):
        pass
