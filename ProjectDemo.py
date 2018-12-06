"""
ProjectDemo.py
Simulates the interaction between a client and the server to demonstrate implemented features.
"""

import signal
import sys
# sys.path.append('..')
import os
import socket
from Client.Client import Client
from Client.ClientUserInterface import ClientUserInterface
from CentralServer.ClientManager import ClientManager


def _get_free_port():
    """
    _get_free_port: Returns a port the OS deems free.
    :return port: <str> A port number provided by the OS.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
    sock.bind(('', 0))
    sock.listen(socket.SOMAXCONN)
    ip, port = sock.getsockname()
    sock.close()
    return port


def handle_interrupt_signal(signal, sig_frame):
    print('ProjectDemo: Closing listening socket and disconnecting from central server.')
    client_manager.disconnect_client(client_id=client_id)
    # terminate the client:
    sys.exit(0)


if __name__ == '__main__':
    # Attach sigint handler:
    signal.signal(signal.SIGINT, handler=handle_interrupt_signal)
    # Instantiate client manager object to handle interfacing with client process:
    client_manager = ClientManager()
    client_listening_port = _get_free_port()
    # Create the client through the ClientManager:
    client_manager_response = client_manager.add_client(
        client_hostname_or_ip=socket.gethostname(),
        client_port=client_listening_port
    )
    print('ProjectDemo [Info]: Client listening on port: %d' % client_listening_port)
    status_code = client_manager_response.split()[0]
    if status_code.upper() == 'OK':
        client_id = int(client_manager_response.split()[1])
    else:
        client_id = None
        print('ProjectDemo [Error]: ClientManager was unable to add the specified client.')
        exit()
    # Try connecting to the server for the handshake:
    client_manager_response = client_manager.connect_client(
        client_id=client_id,
        server_hostname_or_ip='localhost',
        server_port=15014
    )
    client_manager.run_client(client_id=client_id)
    # ON RESUME: Trying to figure out how to tie in the client user-interface. The way it is currently written it is
    # expecting a client instance to associate with, but the clients are supposed to be protected with teh ClientManager
    # so perhaps the ClientManager class should be handed the user interface responsability as well? Perhaps decouple the
    # Client user interface and the client, and instead couple it to the program demo class?.
    # Start the client user-interface:
    # client_user_interface = ClientUserInterface(client_instance=client)
    # client.start()
    # # wait until the client's terminate method is executed:
    # client.join()
    # TODO: Send a sigint signal to this script so it will clean up the sockets.
    # os.kill(os.getpid(), signal.SIGINT)
