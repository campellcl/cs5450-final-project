"""
ClientServerInterface.py
Manages the complexity in communications between the client and the server.
"""

import socket
from Client import Client


class ClientServerInterface:

    server_name_or_ip = None
    server_port = None
    client = None

    def __init__(self, server_name_or_ip, server_port, client):
        self.server_name_or_ip = server_name_or_ip
        self.server_port = server_port
        self.client = client

    def connect(self):
        """
        connect: Sends a message to the central server with the host name of this client and the port the client is
            listening on.
        :return:
        """
        try:
            # Use the central server's listening port to attempt a connection:
            central_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # central_server_socket.bind(('', self.client_port))
            central_server_socket.connect((self.server_name_or_ip, self.server_port))
        except Exception as err:
            print('CentralServerInterface [Error]: Unable to bind client \'%s\' with central server on %s::%s'
                  % (self.client.hostname_or_ip, self.server_name_or_ip, self.server_port))
            return 'BAD\nUnable to reach central server.'

        # Send the connection message and receive the response:
        msg = 'CONNECT\n%s\n' % self.client.id
        central_server_socket.send(msg.encode('utf-8'))
        server_response = central_server_socket.recv(1024).decode('utf-8')
        central_server_socket.close()
        status_code = server_response.split('\n')[0]
        if status_code.upper() == 'BAD':
            return 'BAD\nBad port number. Perhaps it is already in use.'
        else:
            return 'OK\n%s\n' % self.client.id

    def disconnect(self):
        """
        disconnect: Sends a disconnect message to the central server with the ID of the client to remove.
        """
        print('ClientServerInterface [Info]: Received command from the associated Client instance to disconnect(). Attempting to re-establish link with CentralSever.')
        # connect to central server (for graceful termination if possible)
        central_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            central_server_socket.connect((self.server_name_or_ip, self.server_port))
            print('ClientServerInterface [Info]: Link re-established with server. Sending disconnect() command.')
        except Exception as err:
            print('CentralServerInterface [Error]: Unable to reach central server. '
                  'Closing listening socket and aborting connection attempt.')
            central_server_socket.close()
            return 'BAD\nUnable to reach central server'
        # send message and receive response:
        msg = 'DISCONNECT\n%s\n' % self.client.id
        central_server_socket.send(msg.encode('utf-8'))
        response = central_server_socket.recv(1024).decode('utf-8')
        print('CentralServerInterface [Info]: Exchange successful, CentralServer responds '
              'with: %s client %d now disconnected.' % (response.split()[0], self.client.id))
        central_server_socket.close()
        return 'OK\n'

    def post(self, img_name, img):
        """
        post: Sends an image as an array to the central server (for storage purposes).
        :return:
        """
        # connect to central server
        central_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            central_server_socket.connect((self.server_name_or_ip, self.server_port))
        except Exception as err:
            print('CentralServerInterface [Error]: Unable to reach central server. '
                  'Closing listening socket and aborting connection attempt.')
            central_server_socket.close()
            return 'BAD\nUnable to reach central server'
        # send message and receive response
        msg = ('POST\n%s\n' % img_name).encode('utf-8') + img + '\r\n'.encode('utf-8')
        central_server_socket.send(msg)
        response = central_server_socket.recv(1024).decode('utf-8')
        central_server_socket.close()
        return 'OK\n'
