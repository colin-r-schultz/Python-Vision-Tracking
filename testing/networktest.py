import socket
import threading
import time


soc = socket.socket(type=socket.SOCK_DGRAM)
soc.bind(('', 5555))

for i in range(5):
	soc.sendto('join'.encode(), ('', 1922))


running = True


def rcv_loop():
	global soc, running
	while running:
		b = soc.recv(5)
		print(b.decode())


threading.Thread(target=rcv_loop).start()

for i in range(5):
	s = 'test' + str(i)
	soc.sendto(s.encode(), ('', 1922))

for i in range(5):
	s = 'stop' + str(i)
	soc.sendto(s.encode(), ('', 1922))

time.sleep(3)
running = False

