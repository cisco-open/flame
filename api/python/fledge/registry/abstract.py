from abc import abstractmethod


class AbstractRegistryAgent:
    @abstractmethod
    def register(self, job, channel, role, uid, endpoint):
        pass

    @abstractmethod
    def get(self, job, channel):
        pass

    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def close(self):
        pass
