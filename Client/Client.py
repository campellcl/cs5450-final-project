"""
Client.py
Handles the user interface with the central server (CentralServer.py instance) issuing the classifications.
"""

__author__ = 'Chris Campell'
__created__ = '11/29/2018'

import socket
import threading
import os
import signal
from PIL import Image
import sys
sys.path.append('..')
from Client.CentralServerInterface import CentralServerInterface

class Client(threading.Thread):

    central_server_contact = None
    client_listening_port = None
    client_name = None
    client_id = None
    valid_image_extensions = ['.jpg', '.jpeg', '.png']

    def __init__(self, central_server_contact, client_listening_port, client_id):
        threading.Thread.__init__(self)
        self.central_server_contact = central_server_contact
        self.client_listening_port = client_listening_port
        self.client_name = socket.gethostname()
        self.client_id = client_id

    def print_usage_info(self):
        print('\npost <image> - Send an image file to the central server and store it. This image will now be an '
              'applicable target for following commands.')
        print('quit')

    def load_image_tensor(self, img_path):
        img = None
        if os.path.exists(img_path) and os.path.isfile(img_path):
            img_name = os.path.basename(img_path).split('.')[0]
            img_extension = os.path.basename(img_path).split('.')[1]
            if img_extension.lower() in self.valid_image_extensions:
                try:
                    img = Image.open(img_path)
                except IOError as err:
                    print('Client: Could not locate image: \'%s\' at the provided path: \'%s\'. Received error:\n\t%s'
                          % (img_name, img_path, err))
        return img

    def run(self):
        """
        run: This method executes code in a loop until the user enters quit. It displays the prompt, reads the user's
            input, and responds to the user's input.
        """
        while True:
            self.print_usage_info()
            user_input = input()
            split_user_input = user_input.split(' ')
            if split_user_input[0].lower() == 'post':
                if len(split_user_input) > 1:
                    img_name = split_user_input[1]
                    # Load the image:
                    img = self.load_image_tensor(img_path=img_name)
                    # Connect to the server:
                    self.central_server_contact.connect()
                    # TODO: Send the image to the server:
                    self.central_server_contact.disconnect()
            elif split_user_input[0].lower() == 'quit':
                self.central_server_contact.disconnect()
                raise os.kill(os.getpid(), signal.SIGINT)
            else:
                print('Unrecognized command. Malformed input.')




