from base import Model
import tensorflow as tf
import numpy as np


def bilinear_variable():
	weights = np.zeros((8, 8), dtype=np.float32)
	for x in range(8):
		for y in range(8):
			weights[x, y] = (1 - abs(3.5 - x) / 4) * (1 - abs(3.5 - y) / 4)
	weights = weights.reshape((8, 8, 1, 1))
	return tf.Variable(weights)


def create_model():
	inp = tf.placeholder(tf.float32, [None, 240, 320, 3], name='input')
	conv = tf.layers.conv2d(inp, 32, 5, 2, padding='same', activation=tf.nn.relu, name='conv1')
	conv = tf.layers.conv2d(conv, 64, 3, padding='same', activation=tf.nn.relu, name='conv2')
	conv = tf.layers.conv2d(conv, 64, 3, padding='same', activation=tf.nn.relu, name='conv3')
	pool1 = tf.layers.max_pooling2d(conv, 2, 2, name='pool1')
	conv = tf.layers.conv2d(pool1, 64, 3, padding='same', activation=tf.nn.relu, name='conv4')
	conv = tf.layers.conv2d(conv, 64, 3, padding='same', activation=tf.nn.relu, name='conv5')
	pool2 = tf.layers.max_pooling2d(conv, 2, 2, name='pool2')
	conv = tf.layers.conv2d(pool2, 128, 3, padding='same', activation=tf.nn.relu, name='conv6')
	conv = tf.layers.conv2d(conv, 128, 3, padding='same', activation=tf.nn.relu, name='conv7')
	pool3 = tf.layers.max_pooling2d(conv, 2, 2, name='pool3')
	conv = tf.layers.conv2d(pool3, 128, 3, padding='same', activation=tf.nn.relu, name='conv8')
	conv = tf.layers.conv2d(conv, 128, 3, padding='same', activation=tf.nn.relu, name='conv9')
	conv = tf.layers.conv2d(conv, 1, 1, name='conv10')
	size = tf.constant([60, 80], tf.int32)
	resize1 = tf.image.resize_nearest_neighbor(conv, size)
	skip2 = tf.layers.conv2d(pool2, 1, 1, name='skip2')
	resize2 = tf.image.resize_nearest_neighbor(skip2, size)
	skip1 = tf.layers.conv2d(pool1, 1, 1, name='skip1')
	concat = resize1 + resize2 + skip1
	batch_size = tf.shape(inp)[0]
	output = tf.nn.conv2d_transpose(concat, bilinear_variable(), [batch_size, 240, 320, 1], [1, 4, 4, 1])
	output = tf.nn.sigmoid(output, name='output')

	return inp, output


class LegitModel(Model):
	def __init__(self, train=False):
		super().__init__('legitmodel', train, input_size=(240, 320), output_size=(240, 320))

	def build_model(self):
		self.input, self.output = create_model()

if __name__ == '__main__':
	LOAD = False
	model = LegitModel(True)
	if LOAD:
		model.load_model()
	model.train(checkpoint=5, batch_size=64, batches=1000)

