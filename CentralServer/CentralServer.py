"""
CentralServer.py
Listens to connection requests and manages interface between client and the class controlling the classification
    complexity (Classifier.py).
"""

import signal
import socket
import sys
from ClientList import ClientList

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
        client_list = None

        def __init__(self, server_listening_port, server_hostname):
            self.server_listening_port = server_listening_port
            self.server_hostname = server_hostname
            self.client_list = ClientList(clients=None)
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
                client_connection_socket, client_ip_and_port = self.server_listening_socket.accept()
                # Receive the message:
                msg = client_connection_socket.recv(1024)
                response = self.process_message(
                    msg.decode('utf-8'),
                    client_ip=client_ip_and_port[0],
                    client_port=client_ip_and_port[1]
                )
                client_connection_socket.send(response.encode('utf-8'))
                # Close the connection with the client:
                client_connection_socket.close()

        def __str__(self):
            return repr(self) + self.server_listening_port + self.server_hostname

        def close(self):
            self.server_listening_socket.close()

        def _execute_connect(self, client_ip, client_port, client_id):
            """
            _execute_connect: This method is run when the client sends a 'CONNECT\n<client-id>\n' command to the central
                server. The connected client will be added to the server's list of clients if it doesn't already exist
                as a connected client. The OK message is sent to the client if the connection attempt was successful,
                otherwise the BAD message is sent to the client.
            :return:
            """
            result = self.client_list.add_client(client_ip=client_ip, client_port=client_port, client_id=client_id)
            return result

        def _execute_disconnect(self, client_id):
            """
            _execute_disconnect: This method is run when the client sends a 'DISCONNECT\n<client-id>\n' command to the
                central server. The specified client will be removed from the server's list of clients if it hasn't
                already been removed. The OK message is sent to the client if the disconnection attempt was successful,
                otherwise the BAD message is sent to the client.
            :return:
            """
            result = self.client_list.remove_client(client_id)
            return result

        def _execute_post(self):
            """
            _execute_post: This method is run when the client sends a 'POST\n<image_vector>' command to the central
                server. The specified image will be added to the
            :return:
            """

            raise NotImplementedError

        def process_message(self, msg, client_ip, client_port):
            """
            process_message: Responds to the message from the client. The received message must be defined in my protocol.
            :return response: <str> The server's response to the msg received from the client.
            """
            response = None
            # Split on white space:
            words = msg.split()
            if len(words) == 0:
                response = 'BAD\n'
                return response
            if words[0].upper() == 'CONNECT' and len(words) == 2:
                return self._execute_connect(client_ip=client_ip, client_port=client_port, client_id=words[1])
            elif words[0].upper() == 'DISCONNECT' and len(words) == 2:
                return self._execute_disconnect(client_id=words[1])
            elif words[0].upper() == 'POST'and len(words) == 2:
                return self._execute_post()
            else:
                response = 'BAD\n'
                return response

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

    def process_message(self, msg):
        self.central_server.process_message(msg=msg)


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

