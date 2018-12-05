"""
ClientUserInterface.py
Handles the user interface on the client side.
"""

__author__ = 'Chris Campell'
__created__ = '11/29/2018'

import socket
import threading
import os
import signal
from PIL import Image
# import sys
# sys.path.append('..')
from Client.Client import Client
from Client.ClientServerInterface import ClientServerInterface


class ClientUserInterface():
    client_instance = None
    valid_image_extensions = ['.jpg', '.jpeg', '.png']

    def __init__(self, client_instance):
        self.client_instance = client_instance
        self.run()

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
        else:
            print('Client: The provided image: \'%s\' could not be located relative to the current directory. '
                  'Try using the full file path.' % img_path)
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
                    if img is None:
                        continue
                    self.client_instance.post()
                    # Connect to the server:
                    self.central_server_contact.connect()
                    # Send the image to the server:
                    self.central_server_contact.post(img)
                    # Disconnect from the central server:
                    self.central_server_contact.disconnect()
            elif split_user_input[0].lower() == 'quit':
                self.central_server_contact.disconnect()
                raise os.kill(os.getpid(), signal.SIGINT)
            else:
                print('Unrecognized command. Malformed input.')

