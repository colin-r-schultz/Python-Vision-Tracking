import cv2

from base import Input


class CameraInput(Input):
	def __init__(self):
		super().__init__()
		self.cap = cv2.VideoCapture(1)
		self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
		self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

	def get(self):
		_, img = self.cap.read()
		return img
