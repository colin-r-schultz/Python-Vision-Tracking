import tensorflow as tf
import numpy as np
import struct

sess = tf.Session()

inp = tf.placeholder(tf.float32, [None, 4, 4, 3])
conv1 = tf.layers.conv2d(inp, 2, 2, 2, activation=tf.nn.relu, bias_initializer=tf.random_normal_initializer)
conv2 = tf.layers.conv2d(conv1, 1, 2, 1, activation=tf.nn.sigmoid, use_bias=False)

sess.run(tf.global_variables_initializer())

l = list()
s = 0
vs = tf.global_variables()
for v in vs:
	print(v.name)
	print(v.shape)
	va = sess.run(v)
	print(va)
	if len(va.shape) == 4:
		va = va.transpose((3, 0, 1, 2)).reshape(-1)
	s += len(va)
	l.extend(list(va))

test = np.random.normal(0, 1, (1, 4, 4, 3))
print(test)
print(test.reshape((4, 12)))
print('result:')
print(sess.run(conv1, feed_dict={inp: test}))
print(sess.run(conv2, feed_dict={inp: test}))
test = test.reshape(-1)
l = list(test) + l

array = struct.pack('>{}f'.format(len(l)), *l)
file = open('/Users/Family/Desktop/Colin\'s Stuff/GRT/GRT Code/Neural Network/src/main/testweights.192', 'wb')
file.write(array)
file.close()
