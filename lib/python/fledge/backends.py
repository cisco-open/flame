from .backend.local import LocalBackend
from .backend.mqtt import MqttBackend
from .backend.p2p import PointToPointBackend
from .config import BACKEND_TYPE_LOCAL, BACKEND_TYPE_MQTT, BACKEND_TYPE_P2P
from .object_factory import ObjectFactory


class BackendProvider(ObjectFactory):
    def get(self, backend_name, **kargs):
        return self.create(backend_name, **kargs)


backend_provider = BackendProvider()
backend_provider.register(BACKEND_TYPE_LOCAL, LocalBackend)
backend_provider.register(BACKEND_TYPE_P2P, PointToPointBackend)
backend_provider.register(BACKEND_TYPE_MQTT, MqttBackend)
