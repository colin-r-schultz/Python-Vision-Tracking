import numpy as np

from base import PostProcess

xs = np.array(range(20)) + 0.5
ys = np.array(range(15)) + 0.5


class WeightedAverage(PostProcess):
	def run(self, img):
		x = np.average(xs, weights=np.sum(img, 0))
		y = np.average(ys, weights=np.sum(img, 1))
		return np.array([x, y])


class BinaryWeightedAverage(PostProcess):
	def __init__(self, thresh):
		self.t = thresh

	def run(self, img):
		img = np.greater(img, self.t) * img
		xw = np.sum(img, 0)
		yw = np.sum(img, 1)
		if np.sum(xw) == 0 or np.sum(yw) == 0:
			return None
		x = np.average(xs, weights=xw)
		y = np.average(ys, weights=yw)
		return np.array([x, y])


class ThresholdBWA(PostProcess):
	def __init__(self, detect, thresh):
		self.detect = detect
		self.bwa = BinaryWeightedAverage(thresh)

	def run(self, img):
		if np.max(img) < self.detect:
			return None
		return self.bwa.run(img)
