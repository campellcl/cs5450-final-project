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
from Client.ClientUserInterface import ClientUserInterface


class Client:
    listening_port = None
    hostname_or_ip = None
    id = None
    server_contact = None
    images = None
    user_interface = None

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
        print('Client [Info]: Received instructions to execute disconnect() command from the associated ClientUserInterface instance.')
        if self.server_contact is not None:
            print('Client [Info]: This client instance has a pre-existing connection with the server. Notifying the associated ClientServerInterface of the intent to disconnect.')
            self.server_contact.disconnect()
        else:
            print('Client [Error]: Client cannot disconnect, is already disconnected or has lost contact with server.')

    def post(self, img_name, img):
        response = None
        print('Client [Info]: Received instructions to execute post() command from the associated ClientUserInterface instance.')
        if self.server_contact is not None:
            response = self.server_contact.post(img_name=img_name, img=img)
        return response

    def list_command(self, subcommand):
        if subcommand == 'IMAGES':
            sys_out = ''
            image_list = self.server_contact.list_images(client_id=self.id)
            if image_list is not None:
                for i, image in enumerate(image_list):
                    if i != len(image_list) - 1:
                        sys_out = sys_out + '%d) %s\n' % (i, image)
                    else:
                        sys_out = sys_out + '%d) %s' % (i, image)
                print(sys_out)
                return True
            else:
                return False

    def run(self):
        self.user_interface = ClientUserInterface(self)
