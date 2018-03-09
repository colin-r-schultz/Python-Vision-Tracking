from base import Model
import tensorflow as tf


def create_model():
	inp = tf.placeholder(tf.float32, [None, 240, 320, 3], name='input')
	conv = tf.layers.conv2d(inp, 32, 5, 2, padding='same', activation=tf.nn.relu, name='conv1')
	conv = tf.layers.conv2d(conv, 32, 3, padding='same', activation=tf.nn.relu, name='conv2')
	conv = tf.layers.conv2d(conv, 32, 3, padding='same', activation=tf.nn.relu, name='conv3')
	pool1 = tf.layers.max_pooling2d(conv, 2, 2, name='pool1')
	conv = tf.layers.conv2d(pool1, 64, 3, padding='same', activation=tf.nn.relu, name='conv4')
	pool2 = tf.layers.max_pooling2d(conv, 2, 2, name='pool2')
	conv = tf.layers.conv2d(pool2, 64, 3, padding='same', activation=tf.nn.relu, name='conv5')
	conv = tf.layers.conv2d(conv, 128, 3, padding='same', activation=tf.nn.relu, name='conv6')
	pool3 = tf.layers.max_pooling2d(conv, 2, 2, name='pool3')
	conv = tf.layers.conv2d(pool3, 128, 3, padding='same', activation=tf.nn.relu, name='conv7')
	output = tf.layers.conv2d_transpose(conv, 1, 16, 16, activation=tf.nn.sigmoid, use_bias=False, name='output')

	return inp, output


class HighResModel(Model):
	def __init__(self, train=False):
		super().__init__('highresmodel', train, input_size=(240, 320), output_size=(240, 320))

	def build_model(self):
		self.input, self.output = create_model()

if __name__ == '__main__':
	LOAD = True
	model = HighResModel(True)
	if LOAD:
		model.load_model()
	model.train(checkpoint=5, batch_size=64, batches=1000)

