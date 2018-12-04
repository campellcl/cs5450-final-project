"""
Client.py
Client side of the application.
"""

__author__ = 'Chris Campell'
__created__ = '12/3/2018'

import threading
import socket


class Client(threading.Thread):
    listening_port = None
    hostname_or_ip = None
    id = None
    central_server_contact = None
    images = None

    def __init__(self, client_listening_port, client_id):
        threading.Thread.__init__(self)
        self.client_listening_port = client_listening_port
        self.client_name = socket.gethostname()
        self.client_id = client_id
