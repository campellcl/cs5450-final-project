"""
Client.py
Client side of the application.
"""

__author__ = 'Chris Campell'
__created__ = '12/3/2018'

import threading
import socket
import sys
from Client.ClientServerInterface import ClientServerInterface


class Client:
    listening_port = None
    hostname_or_ip = None
    id = None
    server_contact = None
    images = None

    def __init__(self, client_hostname_or_ip, client_listening_port, client_id):
        self.hostname_or_ip = client_hostname_or_ip
        self.listening_port = client_listening_port
        self.id = client_id

    def __eq__(self, other):
        if isinstance(other, Client):
            if self is other:
                return True
            if self.id == other.id:
                if self.listening_port == other.listening_port:
                    return self.hostname_or_ip == other.hostname_or_ip
        return False

    def connect(self, server_hostname_or_ip, server_port):
        if self.server_contact is None:
            self.server_contact = ClientServerInterface(
                server_name_or_ip=server_hostname_or_ip,
                server_port=server_port,
                client=self
            )
        response = self.server_contact.connect()
        status_code = response.split()[0]
        if status_code.upper() == 'BAD':
            print(response[4:])
            print('Exiting')
            sys.exit()
        else:
            return response

    def disconnect(self):
        if self.server_contact is not None:
            self.server_contact.disconnect()
        else:
            print('Client [Error]: Client cannot disconnect, is already disconnected or has lost contact with server.')

    def add_image(self, img_tensor):
        # self.ImageManager.add_image(img_tensor)
        raise NotImplementedError



