import numpy as np
import cv2

src = np.array([[188, 308], [508, 300], [344, 197], [488, 165]], dtype=np.float32)
dst = np.array([[20.8, -12], [20.8, 12], [30, 0], [32.8, 12]], dtype=np.float32)
M = cv2.getPerspectiveTransform(src, dst)
print(M)
v = np.dot(M, np.array([304, 240, 1]))
print(v[0:2]/v[2])
