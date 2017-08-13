import socket

IP = "127.0.0.1"
UDP_PORT = 6000
MSG = "blah"

#SOCK_DGRAM means UDP socket as we don't need complicated logic
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

#sends the data of course
sock.sendto(MSG.encode(), (IP, UDP_PORT))
print('sent{0}'.format(MSG))