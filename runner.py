import cv2
import numpy as np
from base import Display
from camerainput import CameraInput
from client import Client
from postprocessors import BinaryWeightedAverage

from ogmodel import OGModel

source = Client()
model = OGModel()
model.load_model()
display = Display()
post = BinaryWeightedAverage(0.9)


def to_float(img):
	img = img.astype(np.float32)
	img /= 255
	return img

while True:
	img = source.get()
	fimg = to_float(img)
	res = model.run(fimg).reshape((15, 20))
	data = post.run(res)
	source.send(data)
	display.display_input(img)
	display.display_output(res)
	display.display_augmented(img, data)
	k = cv2.waitKey(1)
	if k != -1:
		break
