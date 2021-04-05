from abc import abstractmethod


class AbstractBackend:
    @abstractmethod
    async def _setup_server(self):
        pass

    @abstractmethod
    def uid(self):
        pass

    @abstractmethod
    def endpoint(self):
        pass

    @abstractmethod
    def connect(self, peer, path):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def send(self, peer, msg):
        pass

    @abstractmethod
    def recv(self, peer):
        pass
