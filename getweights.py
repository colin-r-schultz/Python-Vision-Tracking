import tensorflow as tf
import numpy as np
import struct

sess = tf.Session()

inp = tf.placeholder(dtype=tf.float32, shape=[None, 488, 648, 3], name='input')
conv1 = tf.layers.conv2d(inp, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu, name='conv1')
conv2 = tf.layers.conv2d(conv1, filters=32, kernel_size=3, strides=2, activation=tf.nn.relu, name='conv2')
conv3 = tf.layers.conv2d(conv2, filters=16, kernel_size=2, strides=2, activation=tf.nn.relu, name='conv3')
output = tf.layers.conv2d(conv3, filters=1, kernel_size=2, strides=2, activation=tf.nn.sigmoid, use_bias=False, name='output')

saver = tf.train.Saver()

# sess.run(tf.initialize_all_variables())

saver.restore(sess, 'save/save.ckpt')

l = list()
s = 0
vs = tf.global_variables()
for v in vs:
	print(v.name)
	print(v.shape)
	va = sess.run(v)
	if len(va.shape) == 4:
		va = va.transpose((3, 0, 1, 2)).reshape(-1)
	s += len(va)
	l.extend(list(va))

array = struct.pack('>{}f'.format(len(l)), *l)
file = open('weights.192', 'wb')
file.write(array)
file.close()
print(s)
