import tensorflow as tf

from base import Model


def create_model():
	inp = tf.placeholder(dtype=tf.float32, shape=[None, 480, 640, 3], name='input')
	padding = tf.constant([[0, 0], [4, 4], [4, 4], [0, 0]])
	pad = tf.pad(inp, padding, 'CONSTANT', constant_values=0)
	conv1 = tf.layers.conv2d(pad, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu, name='conv1')
	conv2 = tf.layers.conv2d(conv1, filters=32, kernel_size=3, strides=2, activation=tf.nn.relu, name='conv2')
	conv3 = tf.layers.conv2d(conv2, filters=16, kernel_size=2, strides=2, activation=tf.nn.relu, name='conv3')
	output = tf.layers.conv2d(conv3, filters=1, kernel_size=2, strides=2, use_bias=False, activation=tf.nn.sigmoid,
																											name='output')
	return inp, output


class OGModel(Model):
	def __init__(self, train=False):
		super().__init__('ogmodel', train)

	def build_model(self):
		self.input, self.output = create_model()

if __name__ == '__main__':
	model = OGModel(True)
	model.train(checkpoint=5)



