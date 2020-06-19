
import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from ptfrenderer import utils
from ptfrenderer import camera
from ptfrenderer import mesh

import matplotlib.pyplot as plt


def cube():
	# coordinates
	c = [[-1.,-1.,-1], [+1.,-1.,-1], [-1.,+1.,-1], [+1.,+1.,-1],
	     [-1.,-1.,+1], [+1.,-1.,+1], [-1.,+1.,+1], [+1.,+1.,+1]] # corners
	xyz = [[c[0], c[1], c[2], c[3],
	        c[0], c[2], c[4], c[6],
	        c[0], c[4], c[1], c[5],
	        c[7], c[5], c[6], c[4],
	        c[7], c[3], c[5], c[1],
	        c[7], c[6], c[3], c[2]]]

	uv = tf.constant([[[-1., -1.], [+1., -1.], [-1., +1.], [+1, +1]]])
	uv = tf.tile(uv, [1, 6, 1]) # same UV mapping coordinates for all faces of the cube
	triangles = [[0,2,1], [1,2,3],
	             [4,6,5], [5,6,7],
	             [8,10,9], [9,10,11],
	             [12,14,13], [13,14,15],
	             [16,18,17], [17,18,19],
	             [20,22,21], [21,22,23]]
	return tf.constant(xyz), uv, tf.constant(triangles)

def get_camera_params(field_of_view, angles):
	R = camera.euler(angles)
	K, T = camera.look_at_origin(field_of_view)
	K, T, R = utils._broadcast(K, T, R)
	P = tf.matmul(T, camera.hom(R, 'R'))
	return K, P

def checker_board():
	resolution = 256
	idx = tf.range(resolution) // (resolution // 8)
	x = idx[:,tf.newaxis]
	y = idx[tf.newaxis,:]
	v = tf.cast((x+y) % 2, tf.float32)[tf.newaxis,...,tf.newaxis]
	texture = tf.tile(v, [1,1,1,3])
	return texture

xyz, uv, triangles = cube()
xyz *= 0.5
texture = checker_board()

field_of_view = 45. * np.pi/180.
angles = tf.constant([[30., -40., 0.]]) * np.pi/180
K, P = get_camera_params(field_of_view, angles)

imsize = [480, 640]

# The texture can be added via UV mapping
imuva = mesh.render(xyz, uv, triangles, K, P, imsize, margin=0.)
ima = tf.concat(
	[utils.sample2D_bilinear(texture, imuva[...,:2])*imuva[...,2:], imuva[...,2:]],
	-1)
ima = utils.alpha_matte(ima, 0.5) # grey background
im = utils.stack_images(ima[...,:3], [1, 1])

plt.figure(1); plt.imshow(im, interpolation='nearest')
plt.show()

