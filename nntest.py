import cv2
import numpy as np
from highresmodel import HighResModel
from hardcoremodel import HardcoreModel
from legitmodel import LegitModel
from sketchymodel import SketchyModel
import datagenerator
import tensorflow as tf
import math

model = LegitModel()
model.load_model()


def to_float(img):
	img = img.astype(np.float32)
	img /= 255
	return img

while True:
	datagenerator.create_batch(1, (240, 320), (240, 320))
	batch, label = datagenerator.get_batch()
	img = batch[0]
	fimg = cv2.resize(img, (320, 240))
	res = model.run(fimg).reshape((240, 320))
	sm = (cv2.resize(res, (80, 60)) * 255).astype(np.uint8)
	edges = cv2.Canny(sm, 200, 254)
	lines = cv2.HoughLinesP(edges, 1, np.pi / 20, 10, minLineLength=8, maxLineGap=20)
	res = cv2.resize(res, (640, 480)).reshape((480, 640, 1))
	img = cv2.resize(img, (640, 480))
	img[:,:,0:2] -= 0.5 * res * img[:,:,0:2]
	img[:,:,2] += res[:,:,0]
	if lines is not None:
		lines = lines.reshape((-1, 4))
		my = 0
		mi = -1
		for i in range(len(lines)):
			x1, y1, x2, y2 = lines[i]
			if max(y1, y2) > my:
				my = max(y1, y2)
				mi = i
		x1, y1, x2, y2 = lines[mi] * 8
		cv2.line(img, (x1, y1), (x2, y2), (1, 0, 0), 2)
	cv2.imshow('image', img)
	k = cv2.waitKey(0)
	if k == 27:
		break
