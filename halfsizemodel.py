from base import Model
import tensorflow as tf


def create_model():
	inp = tf.placeholder(tf.float32, [None, 240, 320, 3], name='input')
	conv1 = tf.layers.conv2d(inp, 8, 5, 2, padding='same', activation=tf.nn.relu, name='conv1')
	conv2 = tf.layers.conv2d(conv1, 16, 3, padding='same', activation=tf.nn.relu, name='conv2')
	pool1 = tf.layers.max_pooling2d(conv2, 2, 2, name='pool1')
	conv3 = tf.layers.conv2d(pool1, 32, 3, padding='same', activation=tf.nn.relu, name='conv3')
	pool2 = tf.layers.max_pooling2d(conv3, 2, 2, name='pool2')
	conv4 = tf.layers.conv2d(pool2, 32, 3, padding='same', activation=tf.nn.relu, name='conv4')
	conv5 = tf.layers.conv2d(conv4, 16, 3, padding='same', activation=tf.nn.relu, name='conv5')
	pool3 = tf.layers.max_pooling2d(conv5, 2, 2, name='pool3')
	output = tf.layers.conv2d(pool3, 1, 3, padding='same', use_bias=False, activation=tf.nn.sigmoid, name='output')

	return inp, output


class HalfSizeModel(Model):
	def __init__(self, train=False):
		super().__init__('smallmodel', train)
		self.input_size = (240, 320)

	def build_model(self):
		self.input, self.output = create_model()

if __name__ == '__main__':
	LOAD = False
	model = HalfSizeModel(True)
	if LOAD:
		model.load_model()
	model.train(checkpoint=5)

