
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('INFO')
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
	# colors
	red = [1.,0.,0.]; green = [0.,1.,0.]; blue = [0.,0.,1.]
	cyan = [0., 1., 1.]; magenta = [1.,0.,1.]; yellow = [1.,1.,0.]
	color = [[red,red,red,red,
	          green,green,green,green,
	          blue,blue,blue,blue,
	          cyan,cyan,cyan,cyan,
	          magenta,magenta,magenta,magenta,
	          yellow,yellow,yellow,yellow]]
	triangles = [[0,2,1], [1,2,3],
	             [4,6,5], [5,6,7],
	             [8,10,9], [9,10,11],
	             [12,14,13], [13,14,15],
	             [16,18,17], [17,18,19],
	             [20,22,21], [21,22,23]]
	return tf.constant(xyz), tf.constant(color), tf.constant(triangles)

def get_camera_params(field_of_view, angles):
	R = camera.euler(angles)
	K, T = camera.look_at_origin(field_of_view)
	K, T, R = utils._broadcast(K, T, R)
	P = tf.matmul(T, camera.hom(R, 'R'))
	return K, P

xyz, color, triangles = cube()
xyz *= 0.5

field_of_view = 30. * np.pi/180.
grid_size = [3, 7]
angle_range = tf.constant([[-30., -135., 0.],[30., 135., 0.]]) * np.pi/180
angles = utils.grid_angles(grid_size, angle_range)
K, P = get_camera_params(field_of_view, angles)

imsize = [256, 256]

ima = mesh.render(xyz, color, triangles, K, P, imsize)
ima = utils.alpha_matte(ima, 1.0) # white background
im = utils.stack_images(ima[...,:3], grid_size)

plt.figure(1); plt.imshow(im, interpolation='nearest')
plt.show()

