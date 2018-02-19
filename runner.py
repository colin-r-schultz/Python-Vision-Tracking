import cv2
import numpy as np
from base import Display
from camerainput import CameraInput
from client import Client
from postprocessors import ThresholdBWA
from halfsizemodel import HalfSizeModel
from ogmodel import OGModel

source = Client()

model2 = HalfSizeModel()
model2.load_model()
display = Display()
post = ThresholdBWA(0.8, 0.4)


def to_float(img):
	img = img.astype(np.float32)
	img /= 255
	return img

while True:
	img = source.get()
	print(img.shape)
	fimg = to_float(img)
	bimg = cv2.resize(fimg, (640, 480))
	res2 = model2.run(fimg).reshape((15, 20))
	data2 = post.run(res2)
	display.display_output(res2)
	display.display_augmented(bimg, data2)
	source.send(data2)
	k = cv2.waitKey(1)
	if k != -1:
		break
