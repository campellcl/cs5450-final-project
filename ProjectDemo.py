"""
ProjectDemo.py
Simulates the interaction between a client and the server to demonstrate the features present.
"""

import signal
import sys
import os
import socket
from Client.Client import Client
from Client.CentralServerInterface import CentralServerInterface


def get_free_port():
    """
    get_free_port: Returns a port the OS deems free.
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
    central_server_contact.disconnect()
    # terminate the client:
    sys.exit(0)


if __name__ == '__main__':
    # Attach sigint handler:
    signal.signal(signal.SIGINT, handler=handle_interrupt_signal)

    client_listening_port = get_free_port()
    client_ids = [0]
    central_server_contact = CentralServerInterface(
        central_server_name='localhost',
        central_server_port=15014,
        client_port=client_listening_port,
        client_id=client_ids[0]
    )
    client = Client(
        central_server_contact=central_server_contact,
        client_listening_port=client_listening_port,
        client_id=client_ids[0]
    )
    # Try connecting to the central server for the handshake:
    response = central_server_contact.connect()
    status_code = response.split(' ')[0]
    if status_code.upper() == 'BAD':
        print(response[4:])
        print('Exiting')
        sys.exit()
    print('Peer server listening on port: %d' % client_listening_port)
    # Start the client:
    client.start()
    # wait until the client's terminate method is executed:
    client.join()
    # Send a sigint signal to this script so it will clean up the sockets.
    os.kill(os.getpid(), signal.SIGINT)
