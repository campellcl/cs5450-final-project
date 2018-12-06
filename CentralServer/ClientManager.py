"""
ClientManager.py
Singleton class which maintains a list of clients, serves as the container for Client type objects.
"""

import collections
from Client.Client import Client

class ClientManager:
    # Singleton instance:
    client_manager = None

    class __ClientManager:
        """ Private singleton instance """
        clients = None

        def __init__(self, clients=None):
            if clients is not None:
                self.clients = clients
            else:
                self.clients = collections.OrderedDict()

        def _client_in_clients(self, client):
            for c_id, c in self.clients.items():
                if c == client:
                    return True
            return False

        def _client_id_in_clients(self, client_id):
            for c_id, c in self.clients.items():
                if c_id == client_id:
                    return True
            return False

        def _remove_client(self, client):
            response = None
            updated_clients = self.clients.copy()
            for c_id, c in self.clients.items():
                if c == client:
                    updated_clients.pop(c_id)
                    response = 'OK\n'
            if response is None:
                response = 'BAD\nClient does not exist in client container.\n'
            else:
                self.clients = updated_clients
            return response

        def _remove_client_by_id(self, client_id):
            """
            _remove_client: Removes the provided client from the client dict as long as the client is in the dict. If the
                client was not in the dict of clients, this method returns a BAD response. Otherwise, this method returns
                with an affirmative OK response to the method invoker.
            :param client_id:
            :return:
            """
            response = None
            updated_clients = self.clients.copy()
            for c_id, c in self.clients.items():
                if c_id == client_id:
                    updated_clients.pop(c_id)
                    response = 'OK\n'
            if response is None:
                response = 'BAD\nClient with id: \'%s\' does not exist in client container.\n' % client_id
            else:
                self.clients = updated_clients
            return response

        def _add_client(self, client_hostname_or_ip, client_port):
            """
            _add_client: Adds the provided client to the client dict as long as the client is not already in the dict.
                If the client was already in the dict, the BAD response is returned. If the client was added to the dict
                successfully, the OK response is returned to the method invoker.
            :return:
            """
            for client_id, client in self.clients.items():
                if client.hostname_or_ip == client_hostname_or_ip and client.listening_port == client_port:
                    response = 'BAD\nClient already connected with name: \'%s\' and port: \'%s\'' \
                               % (client_hostname_or_ip, client_port)
                    return response

            desired_client_id = len(self.clients)
            client = Client(
                client_hostname_or_ip=client_hostname_or_ip,
                client_listening_port=client_port,
                client_id=desired_client_id
            )
            self.clients[desired_client_id] = client
            response = 'OK\n%s\n' % desired_client_id
            return response

        def _get_client_by_id(self, client_id):
            if self._client_id_in_clients(client_id):
                return self.clients[client_id]
            else:
                return None

        def _connect_client(self, client_id, server_hostname_or_ip, server_port):
            if self._client_id_in_clients(client_id=client_id):
                response = self.clients[client_id].connect(
                    server_hostname_or_ip=server_hostname_or_ip,
                    server_port=server_port
                )
                return response
            else:
                print('ClientManager [Error]: The provided client id: \'%s\' is not in the manager\'s list of clients.'
                      % client_id)
                exit(-1)

        def _disconnect_client(self, client_id):
            if self._client_id_in_clients(client_id=client_id):
                response = self.clients[client_id].disconnect()
                return response
            else:
                print('ClientManager [Error]: The provided client id: \'%s\' is not in the manager\'s list of clients.'
                      % client_id)
                exit(-1)

        def _run_client(self, client_id):
            if self._client_id_in_clients(client_id):
                self.clients[client_id].run()

    def __init__(self):
        if not ClientManager.client_manager:
            ClientManager.client_manager = ClientManager.__ClientManager()
        else:
            print('ClientManager [Info]: Singleton instance of ClientManager already instantiated.')

    def __getattr__(self, item):
        getattr(self.client_manager, name=item)

    def add_client(self, client_hostname_or_ip, client_port):
        response = self.client_manager._add_client(client_hostname_or_ip=client_hostname_or_ip, client_port=client_port)
        return response

    def connect_client(self, client_id, server_hostname_or_ip, server_port):
        response = self.client_manager._connect_client(
            client_id=client_id,
            server_hostname_or_ip=server_hostname_or_ip,
            server_port=server_port
        )
        return response

    def disconnect_client(self, client_id):
        response = self.client_manager._disconnect_client(client_id=client_id)
        return response

    def client_id_in_clients(self, client_id):
        response = self.client_manager._client_id_in_clients(client_id=client_id)
        return response

    def client_in_clients(self, client):
        response = self.client_manager._client_in_clients(client=client)
        return response

    def run_client(self, client_id):
        response = self.client_manager._run_client(client_id)

    # def get_client_by_id(self, client_id):
    #     client = self.client_manager._get_client_by_id(client_id=client_id)
    #     return client
