from base import Model
import tensorflow as tf


def create_model():
	inp = tf.placeholder(tf.float32, [None, 240, 320, 3], name='input')
	conv = tf.layers.conv2d(inp, 32, 5, 2, padding='same', activation=tf.nn.relu, name='conv1')
	conv = tf.layers.conv2d(conv, 64, 3, padding='same', activation=tf.nn.relu, name='conv2')
	conv = tf.layers.conv2d(conv, 64, 3, padding='same', activation=tf.nn.relu, name='conv3')
	pool1 = tf.layers.max_pooling2d(conv, 2, 2, name='pool1')
	conv = tf.layers.conv2d(pool1, 128, 3, padding='same', activation=tf.nn.relu, name='conv4')
	conv = tf.layers.conv2d(conv, 128, 3, padding='same', activation=tf.nn.relu, name='conv5')
	pool2 = tf.layers.max_pooling2d(conv, 2, 2, name='pool2')
	conv = tf.layers.conv2d(pool2, 128, 3, padding='same', activation=tf.nn.relu, name='conv6')
	conv = tf.layers.conv2d(conv, 256, 3, padding='same', activation=tf.nn.relu, name='conv7')
	pool3 = tf.layers.max_pooling2d(conv, 2, 2, name='pool3')
	conv = tf.layers.conv2d(pool3, 256, 3, padding='same', activation=tf.nn.relu, name='conv8')
	conv = tf.layers.conv2d(conv, 64, 3, padding='same', activation=tf.nn.relu, name='conv9')
	resize = tf.image.resize_nearest_neighbor(conv, tf.constant([30, 40], tf.int32))
	skip = tf.layers.conv2d(pool2, 64, 1, 1, activation=tf.nn.relu, name='skip2')
	concat = tf.concat([resize, skip], axis=3)
	resize = tf.image.resize_nearest_neighbor(concat, tf.constant([60, 80], tf.int32))
	skip = tf.layers.conv2d(pool1, 64, 1, 1, activation=tf.nn.relu, name='skip1')
	concat = tf.concat([resize, skip], axis=3)
	conv = tf.layers.conv2d(pool1, 128, 1, 1, activation=tf.nn.relu, name='conv10')
	output = tf.layers.conv2d_transpose(concat, 1, 4, 4, activation=tf.nn.sigmoid, use_bias=False, name='output')

	return inp, output


class HardcoreModel(Model):
	def __init__(self, train=False):
		super().__init__('hardcoremodel', train, input_size=(240, 320), output_size=(240, 320))

	def build_model(self):
		self.input, self.output = create_model()

if __name__ == '__main__':
	LOAD = False
	model = HardcoreModel(True)
	if LOAD:
		model.load_model()
	model.train(checkpoint=5, batch_size=64, batches=1000)

