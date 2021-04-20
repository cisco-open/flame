from .frontend.message.basic import BasicMessage
from .frontend.message.keras import KerasMessage
from .object_factory import ObjectFactory


class MessageProvider(ObjectFactory):
    def get(self, backend_name, **kargs):
        return self.create(backend_name, **kargs)


message_provider = MessageProvider()
message_provider.register('basic', BasicMessage)
message_provider.register('keras', KerasMessage)
