import cv2
import numpy as np
import math


def image_to_float(img):
	img = img.astype(np.float32, copy=False)
	img /= 255
	return img


def rotate_image(mat, angle):
	# Stolen from stackoverflow.com/a/33564950/2119685
	height, width = mat.shape[:2]
	image_center = (width / 2, height / 2)

	rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

	radians = math.radians(angle)
	sin = math.sin(radians)
	cos = math.cos(radians)
	bound_w = int((height * abs(sin)) + (width * abs(cos)))
	bound_h = int((height * abs(cos)) + (width * abs(sin)))

	rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
	rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

	rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
	return rotated_mat

img = cv2.imread('images/image0.png', cv2.IMREAD_COLOR)
cv2.imshow('image', img)

cv2.waitKey(0)
