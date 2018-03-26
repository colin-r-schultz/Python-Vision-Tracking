import tensorflow as tf
from legitmodel import LegitModel
from sketchymodel import SketchyModel

lm = LegitModel()
lm.load_model()

sm = SketchyModel
sm.load_model()

sess = tf.Session()
vs = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='legitmodel'))

for v in vs:
	print(v)
