
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.compat.v1.enable_eager_execution()

from ptfrenderer import utils
from ptfrenderer import camera
from ptfrenderer import uv

import matplotlib.pyplot as plt


resolution = 256
xyz = uv.sphere_init([resolution, resolution])
xyz *= 0.5
vertuv, triangles = uv.sphere([resolution, resolution])
texture = tf.zeros_like(xyz) + 0.75

field_of_view = 5. * np.pi/180.
K, P = camera.look_at_origin(field_of_view)

imsize = [480, 640]

def light(xyz, texture, light_direction):
	norm = uv.normals(xyz)
	lambertian_shading = tf.maximum(
		tf.reduce_sum(norm * light_direction[:,tf.newaxis, tf.newaxis,:], -1, keepdims=True),
		0.)
	texture_lit = texture * lambertian_shading
	return texture_lit
# It is left as an exercise to add ambient and Phong illumination.

light_direction = tf.constant([[-1., -0.5, -1.]])
light_direction = tf.linalg.l2_normalize(light_direction, -1)
texture_lit = light(xyz, texture, light_direction)

ima = uv.render(xyz, texture_lit, vertuv, triangles, K, P, imsize, bcg=0.25)

plt.figure(1); plt.imshow(ima[0,...,:3], interpolation='nearest')
plt.show()

