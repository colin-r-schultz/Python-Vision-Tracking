import numpy as np

from base import PostProcess

xs = np.array(range(20)) + 0.5
ys = np.array(range(15)) + 0.5


class WeightedAverage(PostProcess):
	def run(self, img):
		x = 32 * np.average(xs, weights=np.sum(img, 0))
		y = 32 * np.average(ys, weights=np.sum(img, 1))
		return x, y


class BinaryWeightedAverage(PostProcess):
	def __init__(self, thresh):
		self.t = thresh

	def run(self, img):
		img = np.greater(img, self.t)
		xw = np.sum(img, 0)
		yw = np.sum(img, 1)
		if np.sum(xw) == 0 or np.sum(yw) == 0:
			return None
		x = 32 * np.average(xs, weights=xw)
		y = 32 * np.average(ys, weights=yw)
		return x, y
