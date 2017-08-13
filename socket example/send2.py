import socket

IP = "127.0.0.1"
UDP_PORT = 6000
MSG = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

#send MSG in an infinite loop just for testing what happens when the server recieves a msg and when it stops to recieve
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
while True:
	sock.sendto(MSG.encode(), (IP, UDP_PORT))
print('sent{0}'.format(MSG))