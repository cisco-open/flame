from abc import abstractmethod


# abstract class for serializer/desrializer
class AbstractSerDes:
    @abstractmethod
    def to_bytes(self):
        pass

    @abstractmethod
    def to_json(self):
        pass

    @abstractmethod
    def to_object(self):
        pass
