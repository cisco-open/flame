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
    def connect(self, end_id, endpoint):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def send(self, end_id, msg):
        pass

    @abstractmethod
    def recv(self, end_id):
        pass
