"""
Copyright (C) 2020 Attila Szabo
All rights reserved.
This file is part of the pTFrenderer library and is made available under
the terms of the BSD license (see the LICENSE file).
"""

import tensorflow as tf

from .utils import _broadcast

_div = tf.math.divide_no_nan


def transform(P, xyz):
	"""
	It applies the rigid transformation P on the 3D points xyz.

	Input:
	P: [B, 3, 4] tf.float32, rigid transformation matrix
	xyz: [B, N, 3] ft.float32, 3D points
		Note that, the points are represented as row vectors.
		If they were column vectors, tf.matmul would be enough.

	The first dimension is the batch, which is broadcasted.

	Output:
	xyz: [B, N, 3] ft.float32, transformed 3D points
	"""

	P, xyz = _broadcast(P, xyz)
	xyz = tf.matmul(xyz, P[:,:,:3], transpose_b=True) + tf.reshape(P[:,:,3], [-1,1,3])
	return xyz

def project(K, xyz):
	"""
	It applies perspective projection K on points xyz.

	Input:
	K: [B, 3, 3] tf.float32 intrinsic camera parameters
	xyz: [B, N, 3] ft.float32, 3D points in the camera's frame
	
	The first dimension is the batch, which is broadcasted.

	Output:
	uv: [B, N, 2] tf.float32, 2D projected points
	"""

	K, xyz = _broadcast(K, xyz)
	uvw = tf.matmul(xyz, K, transpose_b=True)
	uv = _div(uvw[:,:,:2], uvw[:,:,2:3])
	return uv

def look_at_origin(fow):
	"""
	This function gives some useful intrinsic and 
	extrinsic camera matrices K and T.
	When you render a unit ball using K and T, you will get
	a circle that tightly fits into the image.

	Input:
	fow: [B] tf.float32, field of views in radians, where 0<fow<pi/2
		for each entry

	Output:
	K: [B, 3, 3] tf.float32, intrinsic camera matrices
		The 3D points [0,-1,f] and [0,1,f] are projected to [0,-1] and [0,1].
		The render will convert the 2D coordinates into pixel coordinates
		depending on the resolution.
	T: [B, 3, 4] tf.float32, rigid transformation matrices
		It moves the origin to the focal plane.
	"""

	f = 1./tf.tan(fow/2.) # focal length
	_0 = tf.zeros_like(f); _1 = tf.ones_like(f)
	K = tf.reshape(tf.stack([ f, _0, _0,
		                     _0,  f, _0,
		                     _0, _0, _1], -1), [-1, 3, 3])
	
	T = tf.reshape(tf.stack([_1, _0, _0, _0,
		                     _0, _1, _0, _0,
		                     _0, _0, _1,  f]), [-1, 3, 4])

	return K, T

def euler(angles, euler_mode="xyz"):
	"""
	It converts euler angles to 3x3 rotation matrices.

	Input:
	angles: [B, 3] tf.float32, angles of rotations along the x,y,z axes
	euler_mode: one of "xyz" or "yxz", it denotes the order of rotations

	Output:
	R: [B, 3, 3] tf.float32 rotation matrix
	"""

	Rx = rotation_matrix(angles[:,0], 0)
	Ry = rotation_matrix(angles[:,1], 1)
	Rz = rotation_matrix(angles[:,2], 2)
	if euler_mode == "xyz":
		R = tf.matmul(Rz, tf.matmul(Ry, Rx))
	if euler_mode == "yxz":
		R = tf.matmul(Rz, tf.matmul(Rx, Ry))
	return R

def rotation_matrix(angle, axis):
	"""
	It computes 3x3 rotation matrix from angle at the specified axis.

	Input:
	angles: [B] tf.float32, angles of rotations
	axis: one of {0,1,2}, the axis of rotation.

	Output:
	R: [B, 3, 3] float32, rotation matrices
	"""

	_0 = tf.zeros_like(angle); _1 = tf.ones_like(angle)
	c = tf.cos(angle); s = tf.sin(angle)
	if axis == 0:
		R = [_1, _0, _0,
		     _0,  c, -s,
		     _0,  s,  c]
	if axis == 1:
		R = [ c, _0, -s,
		     _0, _1, _0,
		      s, _0,  c]
	if axis == 2:
		R = [ c, -s, _0,
		      s,  c, _0,
		     _0, _0, _1]
	R = tf.reshape(tf.stack(R, -1), [-1, 3, 3])
	return R

def hom(X, mode):
	"""
	It converts transformation X (translation, rotation or rigid motion)
	to homodenous form.

	Input:
	X: tf.float32 array, which can be either
		[B, 3] float32, 3D translation vectors
		[B, 3, 3] float32, rotation matrices
		[B, 3, 4] float32, rigid motion matrix
	mode: one of 'T', 'R' or 'P' denoting the options above,
		'T' is for translation
		'R' is for rotation
		'P' is for rigid motion

	Output:
	H: [B, 4, 4] float32, the transformation in homogenous form
	"""

	hom_pad = tf.constant([[[0., 0., 0., 0.],
		                    [0., 0., 0., 0.],
		                    [0., 0., 0., 0.],
		                    [0., 0., 0., 1.]]])
	if mode == 'T':
		X = X[:,:,tf.newaxis]
		padding = [[0, 0], [0, 1], [3, 0]]
	if mode == 'R':
		padding = [[0, 0], [0, 1], [0, 1]]
	if mode == 'P':
		padding = [[0, 0], [0, 1], [0, 0]]
	H = tf.pad(X, padding) + hom_pad
	return H

def quat2rotmat(q):
	"""
	It converts SO(3) quaternion representation q to 3x3 matrices.

	Input:
	q: [B,4] tf.float32, quaternions

	Output:
	R: [B, 3, 3] float32, rotation matrices
	"""

	q = tf.math.l2_normalize(q, -1)
	r = q[:,0]; i = q[:,1]; j = q[:,2]; k = q[:,3]
	R = [1-2*(j*j+k*k),   2*(i*j-k*r),   2*(i*k+j*r),
	       2*(i*j+k*r), 1-2*(i*i+k*k),   2*(j*k-i*r),
	       2*(i*k-j*r),   2*(j*k+i*r), 1-2*(i*i+j*j)]
	R = tf.reshape(tf.stack(R, -1), [-1, 3, 3])
	return R

