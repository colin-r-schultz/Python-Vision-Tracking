import tensorflow as tf
import datagenerator

LOAD = True
OVERWRITE = True

sess = tf.Session()

inp = tf.placeholder(dtype=tf.float32, shape=[None, 488, 648, 3], name='input')
conv1 = tf.layers.conv2d(inp, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu, name='conv1')
conv2 = tf.layers.conv2d(conv1, filters=32, kernel_size=3, strides=2, activation=tf.nn.relu, name='conv2')
conv3 = tf.layers.conv2d(conv2, filters=16, kernel_size=2, strides=2, activation=tf.nn.relu, name='conv3')
output = tf.layers.conv2d(conv3, filters=1, kernel_size=2, strides=2, use_bias=False,activation=tf.nn.sigmoid, name='output')
label_in = tf.placeholder(dtype=tf.float32, shape=[None, 15, 20, 1])
loss = tf.reduce_sum(tf.square(label_in - output), name='loss')

optimize = tf.train.AdamOptimizer(0.001).minimize(loss)

saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

if LOAD:
	saver.restore(sess, 'save/save.ckpt')

EPOCHS = 5
BATCH_SIZE = 32
BATCHES = 100


def save_model():
	if not LOAD or OVERWRITE:
		saver.save(sess, 'save/save.ckpt')
		print('Model saved')

datagenerator.create_batch(BATCH_SIZE)
for i in range(BATCHES):
	datagenerator.block_until_ready()
	batch, label = datagenerator.get_batch()
	datagenerator.create_batch(BATCH_SIZE)
	for j in range(EPOCHS):
		sess.run(optimize, feed_dict={inp: batch, label_in: label})
	print('Completed batch {0} of {1} ({2}%)'.format(i+1, BATCHES, (i+1)*100/BATCHES))
	loss_n = sess.run(loss, feed_dict={inp: batch, label_in: label})
	print('Loss: {}'.format(loss_n.reshape((1,))[0]))
	if (i+1) % 5 == 0:
		save_model()

print('Done')
