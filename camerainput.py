import cv2
from base import Input


def compress_and_decompress(img):
	quality = 90
	params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
	_, comp = cv2.imencode('.jpeg', img, params)
	while comp.size > 60000:
		quality -= 10
		params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
		_, comp = cv2.imencode('.jpeg', img, params)
	img = cv2.imdecode(comp, cv2.IMREAD_COLOR)
	return img


class CameraInput(Input):
	def __init__(self, compress=False):
		super().__init__()
		self.cap = cv2.VideoCapture(0)
		self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
		self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
		self.compress = compress

	def get(self):
		_, img = self.cap.read()
		if self.compress:
			img = compress_and_decompress(img)
		return img
