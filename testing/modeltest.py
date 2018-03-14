import tensorflow as tf
import numpy as np


def bilinear_variable():
	weights = np.zeros((8, 8), dtype=np.float32)
	for x in range(8):
		for y in range(8):
			weights[x, y] = (1 - abs(3.5 - x) / 4) * (1 - abs(3.5 - y) / 4)
	weights = weights.reshape((8, 8, 1, 1))
	return tf.Variable(weights)

sess = tf.Session()

inp = tf.placeholder(tf.float32, [None, 2, 2, 1])
shape = tf.shape(inp)[0]
kernel = bilinear_variable()
conv = tf.nn.conv2d_transpose(inp, kernel, [shape, 8, 8, 1], [1, 4, 4, 1])


sess.run(tf.global_variables_initializer())

x = np.ones((2, 2))
print(sess.run(conv, {inp: x.reshape((1, 2, 2, 1))}).reshape((8, 8)))
