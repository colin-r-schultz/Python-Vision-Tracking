import tensorflow as tf
import cv2
import numpy as np
import datagenerator
import time

sess = tf.Session()

inp = tf.placeholder(dtype=tf.float32, shape=[None, 488, 648, 3], name='input')
conv1 = tf.layers.conv2d(inp, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu, name='conv1')
conv2 = tf.layers.conv2d(conv1, filters=32, kernel_size=3, strides=2, activation=tf.nn.relu, name='conv2')
conv3 = tf.layers.conv2d(conv2, filters=16, kernel_size=2, strides=2, activation=tf.nn.relu, name='conv3')
output = tf.layers.conv2d(conv3, filters=1, kernel_size=2, strides=2, activation=tf.nn.sigmoid, use_bias=False, name='output')

saver = tf.train.Saver()

# sess.run(tf.initialize_all_variables())

saver.restore(sess, 'save/save.ckpt')

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


image = np.zeros((1, 488, 648, 3), dtype=np.uint8)
image[0, 4:484, 4:644, :] = cv2.imread('testimage0.png', cv2.IMREAD_COLOR)
image = image.astype(np.float32)
image[0] /= 255
cv2.imshow('input', image[0])
res = sess.run(output, feed_dict={inp: image})
res = res.reshape((15, 20))
cv2.imshow('output', cv2.resize(res, (640, 480), interpolation=cv2.INTER_NEAREST))
print(res)
cv2.waitKey(0)
exit()
# 25, 10, 150
# 100, 180, 300
scale = 1
while True:
	# Take each frame
	_, frame = cap.read()
	# mask = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	# mask = cv2.inRange(mask, np.array([25, 10, 150]), np.array([100, 180, 300]))
	# cv2.imshow('mask', mask)
	image[0, 4:484, 4:644, :] = frame
	cv2.imshow('input', image[0])
	batch = image.astype(np.float32)
	batch /= 255
	batch *= scale
	print(np.amax(batch, None))
	res = sess.run(output, feed_dict={inp: batch})
	res = res.reshape((15, 20))
	res = cv2.resize(res, (640, 480), interpolation=cv2.INTER_NEAREST)
	cv2.imshow('output', res)
	k = cv2.waitKey(1)
	if k != -1:
		break
