"""
ClientList.py
Maintains a list of clients assocated with the central server (CentralServer.py).
"""

from Client.Client import Client


class ClientList:
    clients = None

    def __init__(self, clients=None):
        if clients is not None:
            self.clients = clients
        else:
            self.clients = []

    def client_in_client_list(self, client_id):
        for client in self.clients:
            c_id = client[2]
            if c_id == client_id:
                return True
        return False

    def add_client(self, client):
        """
        add_client: Adds the provided client to the client list as long as the client is not already in the list. If the
            client was already in the list, the BAD response is returned. If the client was added to the list
            successfully, the OK response is returned to the method invoker.
        :return:
        """
        response = None
        for c in self.clients:
            c_id = c.id
            if c_id == client.id:
                response = 'BAD\nClient already connected?'
                return response
        response = 'OK\n'
        self.clients.append(client)
        return response

    def remove_client(self, client_id):
        """
        remove_client: Removes the provided client from the client list as long as the client is in the list. If the
            client was not in the list of clients, this method returns a BAD response. Otherwise, this method returns
            with an affirmative OK response to the method invoker.
        :param client_id:
        :return:
        """
        if self.client_in_client_list(client_id):
            updated_client_list = self.clients.copy()
            for i, client in enumerate(self.clients):
                c_id = client.id
                if c_id == client_id:
                    updated_client_list.pop(i)
                    continue
            self.clients = updated_client_list
            response = 'OK\n'
            return response
        return 'BAD\nClient not in client list.'
