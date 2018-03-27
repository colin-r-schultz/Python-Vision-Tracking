import socket
import threading
import time


soc = socket.socket(type=socket.SOCK_DGRAM)
soc.bind(('', 5555))

address = 'roborio-192-frc.local'

for i in range(5):
	soc.sendto('join'.encode(), (address, 1922))


running = True


def rcv_loop():
	while running:
		b = soc.recv(5)
		print(str(time.time()) + b.decode())


threading.Thread(target=rcv_loop).start()

for i in range(10):
	s = 'test' + str(i)
	print(str(time.time()) + s)
	soc.sendto(s.encode(), (address, 1922))

for i in range(5):
	s = 'stop' + str(i)
	soc.sendto(s.encode(), (address, 1922))

time.sleep(3)
running = False
