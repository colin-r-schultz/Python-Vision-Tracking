import tensorflow as tf
import datagenerator
import cv2
import numpy as np
import os


class Process:
	def __init__(self):
		pass

	def run(self, img):
		return img


class Model(Process):
	def __init__(self, name, train=False, input_size=(480, 640), output_size=(15, 20)):
		self.name = name
		self.sess = tf.Session()
		self.input = None
		self.output = None
		self.label_in = None
		self.loss = None
		self.optimize = None
		self.input_size = input_size
		self.label_size = output_size
		self.allow_train = train
		with tf.variable_scope(name):
			self.build_model()
			if train:
				self.create_loss()
		self.saver = tf.train.Saver()
		self.sess.run(tf.global_variables_initializer())

	def build_model(self):
		pass

	def create_loss(self):
		self.label_in = tf.placeholder(dtype=tf.float32, shape=[None, self.label_size[0], self.label_size[1], 1])
		# self.loss = tf.reduce_sum(tf.square(self.label_in - self.output), name='loss')
		union = tf.reduce_sum(self.label_in * self.output)
		inter = tf.reduce_sum(self.label_in + self.output) - union
		iou = union / inter
		self.loss = 1.0 - iou
		self.optimize = self.create_optimizer().minimize(self.loss)

	def create_optimizer(self):
		return tf.train.AdamOptimizer(0.0001)

	def save_model(self):
		self.saver.save(self.sess, 'save/' + self.name + '.save')
		os.system('git checkout save')
		os.system('git add save/')
		os.system('git commit -m"Save"')
		os.system('git push')
		print('Model saved')

	def load_model(self):
		self.saver.restore(self.sess, 'save/' + self.name + '.save')

	def run(self, img):
		return self.sess.run(self.output, feed_dict={self.input: [img]})[0]

	def train(self, batches=100, batch_size=32, epochs=5, checkpoint=5):
		if not self.allow_train:
			print('The model was not built to be trained')
			return
		datagenerator.create_batch(batch_size, self.input_size, self.label_size)
		losses = list()
		for i in range(batches):
			batch, label = datagenerator.get_batch()
			datagenerator.create_batch(batch_size, self.input_size, self.label_size)
			for j in range(epochs):
				self.sess.run(self.optimize, feed_dict={self.input: batch, self.label_in: label})
			print('Completed batch {0} of {1} ({2}%)'.format(i + 1, batches, (i + 1) * 100 / batches))
			loss_n = self.sess.run(self.loss, feed_dict={self.input: batch, self.label_in: label})
			losses.append(loss_n)
			print('Loss: {}'.format(loss_n.reshape((1,))[0]))
			if checkpoint != 0 and (i + 1) % checkpoint == 0:
				self.save_model()
		print('Done')
		try:
			from matplotlib import pyplot as plt
			plt.plot(losses)
			plt.show()
		except ImportError:
			print('Requires matplotlib to graph loss')


class Input:
	def __init__(self):
		pass

	def get(self):
		return None

	def send(self, data):
		print(data)


class PostProcess:
	def __init__(self):
		pass

	def run(self, img):
		return 0, 0


class Display:
	def __init__(self):
		pass

	def display_input(self, img):
		cv2.imshow('input', img)

	def display_output(self, img):
		img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_NEAREST)
		cv2.imshow('output', img)

	def display_augmented(self, img, data, name='augmented'):
		img = img.copy()
		if data is not None:
			img = cv2.circle(img, (int(data[0]), int(data[1])), 5, np.array([0, 0, 255.0]), thickness=-1)
		cv2.imshow(name, img)
		return img
