import socket
import cv2
import numpy as np
import tensorflow as tf
import struct

sess = tf.Session()

inp = tf.placeholder(dtype=tf.float32, shape=[None, 480, 640, 3], name='input')
padding = tf.constant([[0, 0], [4, 4], [4, 4], [0, 0]])
pad = tf.pad(inp, pading, 'CONSTANT', constant_values=0)
conv1 = tf.layers.conv2d(pad, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu, name='conv1')
conv2 = tf.layers.conv2d(conv1, filters=32, kernel_size=3, strides=2, activation=tf.nn.relu, name='conv2')
conv3 = tf.layers.conv2d(conv2, filters=16, kernel_size=2, strides=2, activation=tf.nn.relu, name='conv3')
output = tf.layers.conv2d(conv3, filters=1, kernel_size=2, strides=2, activation=tf.nn.sigmoid, use_bias=False, name='output')

saver = tf.train.Saver()

saver.restore(sess, '../save/save.ckpt')

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(('', 5555))
s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 60016)
s.sendto('join'.encode(), ('roborio-192-frc.local', 1920))

xs = np.array(range(20)) + 0.5
ys = np.array(range(15)) + 0.5

while True:
	buf = s.recv(60000)
	ar = np.fromstring(buf, dtype=np.uint8)
	ar = cv2.imdecode(ar, cv2.IMREAD_COLOR)
	cv2.imshow('image', ar)
	ar = ar.astype(np.float32)
	ar /= 255
	res = sess.run(output, feed_dict={inp: [ar]}).reshape((15, 20))
	x = 32 * np.average(xs, weights=np.sum(res, 0))
	y = 32 * np.average(ys, weights=np.sum(res, 1))
	print(x, y)
	send = struct.pack('>2d', x, y)
	s.sendto(send, ('roborio-192-frc.local', 1920))
	res = cv2.resize(res, (640, 480), interpolation=cv2.INTER_NEAREST)
	cv2.imshow('output', res)
	k = cv2.waitKey(1)
	if k != -1:
		break

