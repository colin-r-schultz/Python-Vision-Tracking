import cv2
import numpy as np
from threading import Thread
import random
import os


backgrounds = os.listdir('/Users/Family/PycharmProjects/VisionTracking/backgrounds')
images = os.listdir('/Users/Family/PycharmProjects/VisionTracking/images')


class GeneratorThread(Thread):
	def __init__(self, batch_size, img_size=(480, 640), label_size=(15, 20)):
		Thread.__init__(self)
		self.batch_size = batch_size
		self.img_size = img_size
		self.label_size = label_size
		self.ready = False
		self.batch = None

	def run(self):
		self.batch = generate_batch(self.batch_size, self.img_size, self.label_size)
		self.ready = True

current_generator = None


def generate_batch(batch_size, size=(480, 640), label_size=(15, 20)):
	batch = np.zeros((batch_size, size[0], size[1], 3), dtype=np.float32)
	labels = np.zeros((batch_size, label_size[0], label_size[1], 1), dtype=np.float32)
	for i in range(batch_size):
		generate_img(batch[i], labels[i])
	return batch, labels


def generate_img(mat, label_mat):
	file = random.choice(backgrounds)
	bkg = cv2.imread('backgrounds/' + file, cv2.IMREAD_COLOR)
	bkg = image_to_float(bkg)
	file = random.choice(images)
	img = cv2.imread('images/' + file, cv2.IMREAD_UNCHANGED)
	img = image_to_float(img)
	color_scale = 0.5 + random.random() * 1.0
	img[:, :, :3] *= color_scale
	scale = 0.5 + random.random() * 0.2
	img = cv2.resize(img, None, fx=scale, fy=scale)
	if random.random() < 0.5:
		cv2.flip(img, 0, img)
	theta = random.random() * 360
	img = rotate_image(img, theta)
	h, w, _ = img.shape
	modified_img = np.zeros((960+h, 1280+w, 4), dtype=np.float32)
	modified_img[480:480+h, 640:640+w, :] += img
	h, w, _ = modified_img.shape
	x = rand_normal(320, w-320)
	y = rand_normal(240, h-240)
	img = modified_img[y-240:y+240, x-320:x+320, :]
	modified_img = None
	label = img[:, :, 3]
	img = img[:, :, :3]
	h, w, _ = bkg.shape
	scale = max(800/h, 800/w)
	if scale > 1:
		bkg = cv2.resize(bkg, None, fx=scale, fy=scale)
	h, w, _ = bkg.shape
	x = random.randint(0, w-800)
	y = random.randint(0, h-800)
	bkg = bkg[y:y+800, x:x+800, :]
	theta = random.random() * 360
	M = cv2.getRotationMatrix2D((400, 400), theta, 1)
	bkg = cv2.warpAffine(bkg, M, None)
	bkg = bkg[160:640, 80:720, :]
	if random.random() < 0.5:
		cv2.flip(bkg, 0, bkg)
	color_scale = 0.5 + random.random() * 1.0
	bkg *= color_scale
	mask = label.astype(np.int8)
	mask = 1-mask
	bkg = cv2.bitwise_and(bkg, bkg, mask=mask)
	bkg = np.maximum(bkg, img)
	bkg = np.clip(bkg, 0, 1)
	h, w, _ = label_mat.shape
	label = cv2.resize(label, (w, h), interpolation=cv2.INTER_AREA)
	label = label.reshape((h, w, 1))
	label = np.greater(label, 0.5)
	bkg = compress_and_decompress(bkg)
	h, w, _ = mat.shape
	bkg = cv2.resize(bkg, (w, h))
	mat += bkg
	label_mat += label


def compress_and_decompress(img):
	img = image_to_int(img)
	quality = 90
	params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
	_, comp = cv2.imencode('.jpeg', img, params)
	while comp.size > 60000:
		quality -= 10
		params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
		_, comp = cv2.imencode('.jpeg', img, params)
	img = cv2.imdecode(comp, cv2.IMREAD_COLOR)
	img = image_to_float(img)
	return img


def image_to_float(img):
	img = img.astype(np.float32)
	img /= 255
	return img


def image_to_int(img):
	img *= 255
	img = img.astype(np.uint8)
	return img


def rand_normal(low, high, n=4):
	sum = 0
	r = high - low
	for _ in range(n):
		sum += random.randint(0, r)
	sum = int(sum / 4)
	sum += low
	return sum


def rotate_image(mat, angle):
	# stolen from https://stackoverflow.com/a/37347070

	height, width = mat.shape[:2]
	image_center = (width / 2, height / 2)

	rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

	abs_cos = abs(rotation_mat[0, 0])
	abs_sin = abs(rotation_mat[0, 1])

	bound_w = int(height * abs_sin + width * abs_cos)
	bound_h = int(height * abs_cos + width * abs_sin)

	rotation_mat[0, 2] += bound_w / 2 - image_center[0]
	rotation_mat[1, 2] += bound_h / 2 - image_center[1]

	rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
	return rotated_mat


def block_until_ready():
	global current_generator
	current_generator.join()


def create_batch(batch_size, img_size=(480, 640), label_size=(15, 20)):
	global current_generator
	current_generator = GeneratorThread(batch_size, img_size, label_size)
	current_generator.start()


def get_batch():
	global current_generator
	if current_generator is None:
		return None
	block_until_ready()
	if not current_generator.ready:
		return None
	batch = current_generator.batch
	current_generator = None
	return batch

if __name__ == "__main__":
	while True:
		create_batch(1, (240, 320), (240, 320))
		batch, label = get_batch()
		image = batch[0]
		cv2.imshow('image', image)
		res = cv2.resize(label[0], (320, 240), interpolation=cv2.INTER_NEAREST)
		cv2.imshow('label', res)
		k = cv2.waitKey(0)
		if k == 27:
			cv2.imwrite('input.png', image * 255)
			cv2.imwrite('output.png', res * 255)
			break
