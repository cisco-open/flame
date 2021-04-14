from .frontend.serdes.basic import BasicSerDes
from .object_factory import ObjectFactory


class SerDesProvider(ObjectFactory):
    def get(self, backend_name, **kargs):
        return self.create(backend_name, **kargs)


serdes_provider = SerDesProvider()
serdes_provider.register('basic', BasicSerDes)
