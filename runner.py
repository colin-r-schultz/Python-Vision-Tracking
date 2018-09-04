import cv2
import numpy as np
from base import Display
from camerainput import CameraInput
from client import Client
from legitmodel import LegitModel
import math

source = CameraInput()

model = LegitModel()
model.load_model()
display = Display()

src = np.array([[424, 143], [185, 146], [173, 245], [439, 242]], dtype=np.float32)
dst = np.array([[50, 12], [50, -12], [38, -12], [38, 12]], dtype=np.float32)
dst *= 0.0254
M = cv2.getPerspectiveTransform(src, dst)
print(M)


def to_float(img):
	img = img.astype(np.float32)
	img /= 255
	return img

while True:
	img = source.get()
	h, w, _ = img.shape
	img = cv2.resize(to_float(img), (320, 240))
	res = model.run(img).reshape((240, 320))
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
	k = cv2.waitKey(1)
	if k != -1:
		# cv2.imwrite('input.png', img)
		# cv2.imwrite('output.png', cv2.resize(res2 * 255, (640, 480), interpolation=cv2.INTER_NEAREST))
		# cv2.imwrite('augmented.png', aimg)
		break
