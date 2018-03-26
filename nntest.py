import cv2
import numpy as np
from highresmodel import HighResModel
from hardcoremodel import HardcoreModel
from legitmodel import LegitModel
from sketchymodel import SketchyModel
import datagenerator
import tensorflow as tf
import math

model = SketchyModel()
model.load_model()

while True:
	datagenerator.create_batch(1, (240, 320), (240, 320))
	batch, label = datagenerator.get_batch()
	image = batch[0]
	cv2.imshow('image', image)
	label = cv2.resize(label[0], (320, 240), interpolation=cv2.INTER_NEAREST)
	ilabel = (label * 255).astype(np.uint8)
	edges = cv2.Canny(ilabel, 200, 255)
	lines = cv2.HoughLinesP(edges, 1, np.pi/180, 10, maxLineGap=15)
	cv2.imshow('label edges', edges)
	ilabel = cv2.cvtColor(ilabel, cv2.COLOR_GRAY2BGR)
	lines = lines.reshape([-1, 4])
	my = 0
	mi = None
	for i in range(len(lines)):
		x1, y1, x2, y2 = lines[i]
		if max(y1, y2) > my:
			my = max(y1, y2)
			mi = i
		ilabel = cv2.line(ilabel, (x1, y1), (x2, y2), (0, 255, 0), 2)
	x1, y1, x2, y2 = lines[mi]
	ilabel = cv2.line(ilabel, (x1, y1), (x2, y2), (0, 0, 255), 2)
	print(math.atan((y1-y2)/(x1-x2)))
	cv2.imshow('label', ilabel)
	res = model.run(image).reshape((60, 80))
	# res2 = cv2.resize(res, (80, 60))
	ires = (res * 255).astype(np.uint8)
	edges = cv2.Canny(ires, 200, 255)
	cv2.imshow('res edges', edges)
	lines = cv2.HoughLinesP(edges, 1, np.pi/180, 5, minLineLength=5, maxLineGap=7)
	lines = lines.reshape([-1, 4])
	res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
	my = 0
	mi = None
	for i in range(len(lines)):
		x1, y1, x2, y2 = lines[i]
		if max(y1, y2) > my:
			my = max(y1, y2)
			mi = i
		res = cv2.line(res, (x1, y1), (x2, y2), (0, 255, 0), 1)
	x1, y1, x2, y2 = lines[mi]
	res = cv2.line(res, (x1, y1), (x2, y2), (0, 0, 255), 1)
	print(math.atan((y1 - y2) / (x1 - x2)))
	cv2.imshow('result', res)
	k = cv2.waitKey(0)
	if k == 27:
		break
