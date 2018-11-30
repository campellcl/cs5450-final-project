"""
ClientList.py
Maintains a list of clients assocated with the central server (CentralServer.py).
"""


class ClientList:
    clients = None

    def __init__(self, clients=None):
        if clients is not None:
            self.clients = clients
        else:
            self.clients = []
