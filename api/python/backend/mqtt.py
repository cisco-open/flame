from .abstract import AbstractBackend


class MqttBackend(AbstractBackend):
    async def _setup_server(self):
        pass

    def uid(self):
        pass

    def endpoint(self):
        pass

    def connect(self, peer, path):
        pass

    def close(self):
        pass

    def send(self, peer, msg):
        pass

    def recv(self, peer):
        pass
