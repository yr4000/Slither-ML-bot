import socket
import time

IP = "127.0.0.1"
UDP_PORT = 6000

#UDP port
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((IP, UDP_PORT))

#sets the max timeout we will wait for a msg
sock.settimeout(0.01)


#the real code has to use call the function "read_observation" which will use try,except:
#try to read infinity msgs. and when there is an exception that means we read all the msgs.
#we want to keep just the last one and return it
while True:
	#try to read a msg. 1024 Bytes at most
	data,addr = sock.recvfrom(1024)