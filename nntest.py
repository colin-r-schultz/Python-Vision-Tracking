import cv2
import numpy as np
from highresmodel import HighResModel
import datagenerator

model = HighResModel()
model.load_model()

while True:
	datagenerator.create_batch(1, (240, 320), (240, 320))
	batch, label = datagenerator.get_batch()
	image = batch[0]
	cv2.imshow('image', image)
	label = cv2.resize(label[0], (320, 240), interpolation=cv2.INTER_NEAREST)
	cv2.imshow('label', label)
	res = model.run(image).reshape((240, 320))
	cv2.imshow('result', res)
	inter = np.sum(res * label)
	union = np.sum(res + label) - inter
	iou = inter/union
	print(1 - iou)
	k = cv2.waitKey(0)
	if k == 27:
		break
