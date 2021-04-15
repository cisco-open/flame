class ObjectFactory(object):
    def __init__(self):
        self._objects = {}

    def register(self, key, obj):
        self._objects[key] = obj

    def create(self, key, **kwargs):
        obj = self._objects.get(key)
        if not obj:
            raise ValueError(key)
        return obj(**kwargs)
