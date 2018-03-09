import cv2
import numpy as np
from base import Display
from camerainput import CameraInput
from client import Client
from postprocessors import ThresholdBWA
from halfsizemodel import HalfSizeModel
from ogmodel import OGModel

source = CameraInput()#Client()

model = HalfSizeModel()
model.load_model()
display = Display()
post = ThresholdBWA(0.8, 0.4)


def to_float(img):
	img = img.astype(np.float32)
	img /= 255
	return img

while True:
	img = source.get()
	h, w, _ = img.shape
	fimg = cv2.resize(to_float(img), (320, 240))
	res = model.run(fimg).reshape((15, 20))
	data = post.run(res)
	data *= w / 20
	display.display_input(img)
	display.display_output(res)
	aimg = display.display_augmented(img, data)
	source.send(data)
	k = cv2.waitKey(1)
	if k != -1:
		# cv2.imwrite('input.png', img)
		# cv2.imwrite('output.png', cv2.resize(res2 * 255, (640, 480), interpolation=cv2.INTER_NEAREST))
		# cv2.imwrite('augmented.png', aimg)
		break
