
import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from ptfrenderer import utils
from ptfrenderer import camera
from ptfrenderer import uv

import matplotlib.pyplot as plt


def get_camera_params(field_of_view, angles):
	R = camera.euler(angles)
	K, T = camera.look_at_origin(field_of_view)
	K, T, R = utils._broadcast(K, T, R)
	P = tf.matmul(T, camera.hom(R, 'R'))
	return K, P

def checker_board(resolution):
	idx = tf.range(resolution) // (resolution // 8)
	x = idx[:,tf.newaxis]
	y = idx[tf.newaxis,:]
	v = tf.cast((x+y) % 2, tf.float32)[tf.newaxis,...,tf.newaxis]
	texture = tf.tile(v, [1,1,1,3])
	return texture

resolution = 256
xyz = uv.sphere_init([resolution, resolution])
xyz *= 0.5
vertuv, triangles = uv.sphere([resolution, resolution])
texture = checker_board(resolution)

field_of_view = 30. * np.pi/180.
angles = tf.constant([[30., -40., 0.]]) * np.pi/180
K, P = get_camera_params(field_of_view, angles)

imsize = [480, 640]

ima = uv.render(xyz, texture, vertuv, triangles, K, P, imsize, bcg=0.5)
im = utils.stack_images(ima[...,:3], [1, 1])

plt.figure(1); plt.imshow(im, interpolation='nearest')
plt.show()

