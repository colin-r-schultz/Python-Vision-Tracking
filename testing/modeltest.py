import tensorflow as tf
import numpy as np

sess = tf.Session()

inp = tf.placeholder(tf.float32, [None, 240, 320, 1])
conv1 = tf.layers.conv2d(inp, 1, 5, 2, padding='same')

sess.run(tf.global_variables_initializer())

print(sess.run(conv1, {inp: np.ones((1, 240, 320, 1))}).shape)
