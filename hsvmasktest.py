import cv2
import numpy as np

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

kernel = np.zeros((3, 3))
kernel.fill(1/9)

while True:

	# Take each frame
	_, frame = cap.read()

	cv2.imshow('image', frame)
	rgb = cv2.inRange(frame, np.array([0, 250, 250]), np.array([255, 255, 255]))
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	hsv = cv2.inRange(frame, np.array([29, 30, 170]), np.array([90, 160, 300]))
	res = hsv+rgb
	cv2.imshow('res', res)
	res =np.greater(cv2.filter2D(res, -1, kernel), 254).astype(np.uint8) * 255
	cv2.imshow('filter', res)
	k = cv2.waitKey(100)
	if k != -1:
		break

cap.release()
cv2.destroyAllWindows()
