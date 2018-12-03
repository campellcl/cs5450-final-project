"""
CentralServerInterface.py
Provides an interface between the client (Client.py) and the central server (CentralServer.py) that encapsulates the
    complexity associated with communicating between the two objects.
"""

__author__ = 'Chris Campell'
__created__ = '11/29/2018'

import socket


class CentralServerInterface:

    central_server_name = None
    central_server_port = None
    client_name = None
    client_port = None
    client_id = None

    def __init__(self, central_server_name, central_server_port, client_port, client_id):
        self.central_server_name = central_server_name
        self.central_server_port = central_server_port
        self.client_name = socket.gethostname()
        self.client_port = client_port
        self.client_id = client_id

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
            central_server_socket.connect((self.central_server_name, self.central_server_port))
        except Exception as err:
            print('CentralServerInterface [Error]: Unable to bind client \'%s\' with central server on %s::%s'
                  % (self.client_name, self.central_server_name, self.central_server_port))
            return 'BAD\nUnable to reach central server.'

        # Send the connection message and receive the response:
        msg = 'CONNECT\n%s\n' % self.client_id
        central_server_socket.send(msg.encode('utf-8'))
        response = central_server_socket.recv(1024).decode('utf-8')
        central_server_socket.close()
        status_code = response.split('\n')[0]
        if status_code.upper() == 'BAD':
            return 'BAD\nBad port number. Perhaps it is already in use.'
        else:
            return 'OK\n'

    def disconnect(self):
        """
        disconnect: Sends a disconnect message to the central server with the ID of the client to remove.
        """
        # connect to central server (for graceful termination if possible)
        central_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            central_server_socket.connect((self.central_server_name, self.central_server_port))
        except Exception as err:
            print('CentralServerInterface [Error]: Unable to reach central server. '
                  'Closing listening socket and aborting connection attempt.')
            central_server_socket.close()
            return 'BAD\nUnable to reach central server'
        # send message and receive response:
        msg = 'DISCONNECT\n%s\n' % self.client_id
        central_server_socket.send(msg.encode('utf-8'))
        response = central_server_socket.recv(1024).decode('utf-8')
        central_server_socket.close()
        return 'OK\n'

    def post(self, img):
        """
        post: Sends an image as an array to the central server (for storage purposes).
        :return:
        """
        # connect to central server
        central_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            central_server_socket.connect((self.central_server_name, self.central_server_port))
        except Exception as err:
            print('CentralServerInterface [Error]: Unable to reach central server. '
                  'Closing listening socket and aborting connection attempt.')
            central_server_socket.close()
            return 'BAD\nUnable to reach central server'
        # send message and receive response
        msg = 'POST\n%s' % img
        central_server_socket.send(msg.encode('utf-8'))
        response = central_server_socket.recv(1024).decode('utf-8')
        central_server_socket.close()
        return 'OK\n'
