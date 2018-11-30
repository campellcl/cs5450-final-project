"""
CentralServer.py
Listens to connection requests and manages interface between client and the class controlling the classification
    complexity (Classifier.py).
"""

import signal
import socket
import sys

__author__ = 'Chris Campell'
__created__ = '11/29/2018'


class CentralServer:
    # Singleton instance:
    central_server = None

    class __CentralServer:
        # Private singleton instance
        server_listening_port = None
        server_hostname = None
        server_listening_socket = None

        def __init__(self, server_listening_port, server_hostname):
            self.server_listening_port = server_listening_port
            self.server_hostname = server_hostname
            # Instantiate and start up the listening socket:
            self.server_listening_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_listening_socket.bind((self.server_hostname, self.server_listening_port))
            # Param backlog=1
            self.server_listening_socket.listen(1)
            print('CentralServer: The CentralServer instance is listening on port %s' % self.server_listening_port)
            # Only responsibility is to accept connection requests:
            self.accept_connection_requests()

        def accept_connection_requests(self):
            while True:
                # Accept the connection request:
                client_connection_socket, client_ip = self.server_listening_socket.accept()
                # Receive the message:
                msg = client_connection_socket.recv(1024)
                # TODO: Build and send the server's response:
                # response = processMessage(msg.decode('utf-8'), peers, addr);
                # client_connection_socket.send(response.encode('utf-8'))

                # Close the connection with the client:
                client_connection_socket.close()


        def __str__(self):
            return repr(self) + self.server_listening_port + self.server_hostname

        def close(self):
            self.server_listening_socket.close()

    def __init__(self, server_listening_port, server_hostname):
        if not CentralServer.central_server:
            CentralServer.central_server = CentralServer.__CentralServer(server_listening_port, server_hostname)
        else:
            # Singleton already instantiated:
            CentralServer.central_server.server_listening_port = server_listening_port
            CentralServer.central_server.server_hostname = server_hostname

    def __getattr__(self, name):
        return getattr(self.central_server, name=name)

    def close(self):
        self.central_server.close()


def handle_interrupt_signal(signal, sig_frame):
    print('CentralServer: Closing listening socket and terminating.')
    central_server.close_socket()
    sys.exit(0)


if __name__ == '__main__':
    # Attach ctrl+c signal handler for server termination:
    signal.signal(signalnum=signal.SIGINT, handler=handle_interrupt_signal)
    # Create the listening socket for connection requests:
    server_port = 15014
    server_name = 'localhost'
    central_server = CentralServer(server_listening_port=server_port, server_hostname=server_name)

