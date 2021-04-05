class Channel(object):
    def __init__(self, name, backend):
        self.name = name
        self.backend = backend

        # a set for storing peers
        self.peers = set()

    def add(self, peer):
        self.peers.add(peer)

    def peers(self):
        return self.peers

    def broadcast(self, msg):
        for peer in self.peers:
            self.send(peer, msg)

    def send(self, peer, msg):
        pass

    def recv(self, peer):
        pass
