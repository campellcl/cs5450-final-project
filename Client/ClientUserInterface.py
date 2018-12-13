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
import numpy as np
from PIL import Image
# import sys
# sys.path.append('..')
from Client import Client
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
        print('list images - Display a list of the images this client has stored on the central server.')
        print('classify <image_index> - Select an image from the list of images this client has stored on the '
              'central server to perform a classification of.')
        print('quit')

    def load_image(self, img_path):
        img = None
        if os.path.exists(img_path) and os.path.isfile(img_path):
            img_name = os.path.basename(img_path).split('.')[0]
            img_extension = '.' + str(os.path.basename(img_path).split('.')[1])
            if img_extension.lower() in self.valid_image_extensions:
                try:
                    img_bin = open(img_path, 'rb')
                    img = img_bin.read()
                    img_bin.close()
                    # pil_img = Image.open(img_path)
                    # img = np.array(pil_img)
                    # pil_img.close()
                except Exception as err:
                    print('Client: Could not locate image: \'%s\' at the provided path: \'%s\'. Received error:\n\t%s'
                          % (img_name, img_path, err))
                return img

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
            if split_user_input[0].upper() == 'POST':
                print('ClientUserInterface [Info]: Recognized POST command, relaying to client instance.')
                if len(split_user_input) > 1:
                    img_path = split_user_input[1]
                    img_name = os.path.basename(img_path)
                    # Load the image:
                    img = self.load_image(img_path=img_path)
                    if img is None:
                        continue
                    response = self.client_instance.post(img_name=img_name, img=img)
                    status_code = response.split()[0]
                    if status_code.upper() == 'OK':
                        print('Client [Info]: Received OK \'%s\' response from server.' % img_name)
            elif split_user_input[0].upper() == 'QUIT':
                print('ClientUserInterface [Info]: Recognized QUIT command, relaying to client instance.')
                self.client_instance.disconnect()
                raise os.kill(os.getpid(), signal.SIGINT)
            elif split_user_input[0].upper() == 'LIST':
                print('ClientUserInterface [Info]: Recognized LIST command, relaying to client instance.')
                subcommand = split_user_input[1].upper()
                self.client_instance.list_command(subcommand=subcommand)
            elif split_user_input[0].upper() == 'CLASSIFY':
                if len(split_user_input) != 2:
                    print('ClientUserInterface [Error]: The CLASSIFY command must be issued along with the index of '
                          'the image which should be classified.')
                else:
                    print('ClientUserInterface [Info]: Recognized CLASSIFY command, relaying to client instance.')
                    server_image_index = int(split_user_input[1])
                    self.client_instance.classify_command(server_image_index=server_image_index)
            else:
                print('Unrecognized command. Malformed input.')

