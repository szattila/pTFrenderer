
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.compat.v1.enable_eager_execution()

from ptfrenderer import utils
from ptfrenderer import camera
from ptfrenderer import mesh
import json

import matplotlib.pyplot as plt
import trimesh


filename = "bunny/reconstruction/bun_zipper.ply"

def align_bunny(xyz):
	xyz_min = tf.reduce_min(xyz, axis=[0,1], keepdims=True)
	xyz_max = tf.reduce_max(xyz, axis=[0,1], keepdims=True)
	xyz = xyz-xyz_min - (xyz_max-xyz_min) / 2.
	r = tf.sqrt(tf.reduce_max(tf.reduce_sum(xyz**2, -1)))
	xyz /= r
	angles = tf.constant([[-180., 0., 0.]]) * np.pi / 180.
	R = camera.hom(camera.euler(angles), 'R')[:,:3,:]
	xyz = camera.transform(R, xyz)
	return xyz

def load_bunny(filename):
	bunny = trimesh.load(filename)
	xyz = tf.constant(bunny.vertices, tf.float32)[tf.newaxis,:,:]
	xyz = align_bunny(xyz)
	triangles = tf.constant(bunny.faces)
	fur_color = tf.constant([[[201., 156., 122.]]]) / 255.
	s = 0.05
	rgb = tf.random.uniform(xyz.shape) * s + (1.-s)*fur_color
	return xyz, rgb, triangles

def light(xyz, rgb, triangles, lights_direction, ambient):
	norm = mesh.vert_normals(xyz, triangles)
	lambertian_shading = tf.maximum(
		tf.reduce_sum(norm * light_direction, -1, keepdims=True),
		0.)
	rgb_lit = rgb * (lambertian_shading + ambient)
	return rgb_lit

xyz, rgb, triangles = load_bunny(filename)
light_direction = tf.constant([[-1., -0.5, -1.]])
light_direction = tf.linalg.l2_normalize(light_direction, -1)
ambient = 0.3
rgb_lit = light(xyz, rgb, triangles, (1.-ambient)*light_direction, ambient)

imsize = [256, 256]

grid_size = [1, 3]
angle_range = tf.constant([[0., -60., 0.], [0., 60., 0.]]) * np.pi / 180.
angles = utils.grid_angles(grid_size, angle_range)
R = camera.hom(camera.euler(angles), 'R')
K, T = camera.look_at_origin(5.*np.pi/180.)
T, R = utils._broadcast(T, R)
P = tf.matmul(T, R)
    
ima = mesh.render(xyz, rgb_lit, triangles, K, P, imsize)
ima = utils.alpha_matte(ima, 1.0) # white background

plt.figure(1); plt.imshow(utils.stack_images(ima[...,:-1], grid_size), interpolation='nearest')
plt.show()

