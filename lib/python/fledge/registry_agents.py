from .object_factory import ObjectFactory
from .registry.etcd_agent import EtcdRegistryAgent
from .registry.local_agent import LocalRegistryAgent


class RegsitryAgentProvider(ObjectFactory):
    def get(self, agent_name, **kargs):
        return self.create(agent_name, **kargs)


registry_agent_provider = RegsitryAgentProvider()
registry_agent_provider.register('local', LocalRegistryAgent)
registry_agent_provider.register('etcd', EtcdRegistryAgent)
