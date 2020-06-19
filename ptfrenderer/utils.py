"""
Copyright (C) 2020 Attila Szabo
All rights reserved.
This file is part of the pTFrenderer library and is made available under
the terms of the BSD license (see the LICENSE file).
"""

import numpy as np
import tensorflow as tf


def _i64(x): return tf.cast(x, tf.int64)
def _f32(x): return tf.cast(x, tf.float32)

def _shape(t):
	"""
	This function helps with tensor shapes conventions.
	In tensorflow 1 it is (was) very useful.

	Input:
	t: tf.Tensor, np.ndarray or None

	Output:
	The shape of the object t in a list
		Each element is either an integer (when the dimension is known) or a
		scalar tensor (when it is partially known). This achieves the following:
			1) Your code will not chrash other codes that rely on known shapes.
			2) Your code can accept dynamic shape tensors, you can use tf.reshape,
				tf.tile etc. with the output of this function
			3) N, C, H, W = _shape(my_image_batch) just works
		If even your tensors rank is unknown, you did not win, but you are welcome
		to figure out a _shape++ function for that scenario.
	"""

	# deal with None (and defer the problem when t=None should not happen)
	if t is None: return None

	# deal with numpy
	if type(t) is np.ndarray:
		s = t.shape
		return s

	# unknown shape
	if t.shape is None: return None

	# shape is known or partially known
	s_dynamic = tf.shape(t)
	s_static = t.shape.as_list()
	def dimsize(i):
		if s_static[i] is not None:
			si = s_static[i]
		else:
			si = s_dynamic[i]
		return si
	s = [dimsize(i) for i in range(len(s_static))]
	return s

def _broadcast(*ins):
	"""
	It broadcast the input argument along the first dimension.

	Example:
	x, y, z = _broadcast(x, y, z)
	"""

	def bsize(x):
		s = _shape(x)
		if s is None: return 0
		return s[0]
	B = tf.reduce_max(tf.stack([bsize(x) for x in ins]))

	def mytile(x):
		s = _shape(x)
		if s is None: return None
		t = [1] * len(s); t[0] = B // s[0]
		xt = tf.tile(x, t)
		return xt
	outs = [mytile(x) for x in ins]
	return outs 

def sample2D_bilinear(im, uv):
	"""
	It samples the points uv from the image im, where uv coordinates
	are in clip-space.

	Input:
	im: [B,H1,W1,C] or [B,H1,W1] tf.float32, the image to be sampled
	uv: [B,N1,N2,...Nm,2] tf.float32, u, v coordinates of sampling locations
		- The top row of pixels in im have coordinates [...,-1] and
		  the bottom row have [...,+1].
		  The coordinates of the center of im are [0, 0]
		  Thus, for rectangular images [-1,-1] is the center of the
		  top-left pixel and [+1,+1] is the center of the bottom-right pixel.
		  This convention s naturally compatible with the clip-space
		  of the mesh.render function.
		- There are arbitrary number of dimensions besides the first (batch)
		  and the last (coordinates). Thus, sampling list of points or
		  resampling images are both supported.
	The first dimension is the batch and the inputs are broadcasted along it.

	Output:
	imuv: [B,N1,N2,...Nm,C] or [B,N1,N2,...Nm] tf.float32, the sampled points
	"""

	im, uv = _broadcast(im, uv)
	sim = _shape(im)
	B, H1, W1 = sim[:3]
	suv = _shape(uv)

	im = tf.reshape(im, [B, H1, W1, -1])

	H1_ = tf.cast(H1, tf.float32); W1_ = tf.cast(W1, tf.float32)
	uv = tf.reshape(uv, [B,-1,2])
	vupix = tf.stack([(H1_-1)*(uv[:,:,1]+1)/2.0,
		              (H1_-1)*(uv[:,:,0]+1)/2.0 + (W1_-H1_)/2.0,], -1)

	imvu = sample2D_bilinear_pix(im, vupix)
	imvu = tf.reshape(imvu, [*suv[:-1], *sim[3:]])
	return imvu

def sample2D_bilinear_pix(im, vu):
	"""
	It samples the points uv from the image im, where uv coordinates
	correspond to pixels.
	Note: Using sample2D_bilinear is more convenient.

	Input:
	im: [B,H1,W1,C] tf.float32, the image to be sampled
	vu: [B,N,2] tf.float32, the sampling locations (in pixel coordinates)
		- The convention is different than usual vu[:,:,0] is the horizontal
		  and vu[:,:,1] is the vertical coordinate. This convention was
		  choosen to avoid transposing.
		- When the sampling locations are out of bound, they are sampled
		  from the closest location of the image.
	The first dimension is the batch and the inputs are broadcasted along it.

	Output:
	imvu: [B,N,C] tf.float32, the sampled points
	"""

	im, vu = _broadcast(im, vu)

	B, H, W, C = _shape(im)
	N = _shape(vu)[-2]

	vu_floor = tf.floor(vu)
	w = vu - vu_floor # [B,N,2]

	vu__ = tf.reshape(vu_floor, [B, N, 1, 2]) + [[[[0,0],[0,1],[1,0],[1,1]]]]
	vu__ = tf.maximum(tf.minimum(vu__, [[[[H-1, W-1]]]]), 0) # [B, N, 4, 2]
	vu__ = tf.reshape(_i64(vu__), [B, N*4, 2])
	imvu__ = tf.gather_nd(im, vu__, batch_dims=1) # [B, N*4, C]
	imvu__ = tf.reshape(imvu__, [B, N, 2, 2, C])
	wv = tf.reshape(tf.stack([1.0-w[:,:,0], w[:,:,0]], -1), [B,N,2,1,1])
	imvu_ = tf.reduce_sum(imvu__* wv, axis=-3) # [B, N, 2, C]
	wu = tf.reshape(tf.stack([1.0-w[:,:,1], w[:,:,1]], -1), [B,N,2,1])
	imvu = tf.reduce_sum(imvu_* wu, axis=-2) # [B, N, C]
	return imvu

def grid_angles(grid_size, angle_range=tf.constant([[0.,0.,0.],[1.,1.,1.]])):
	"""
	I creates a grid of uniformly spaced viewpoints,

	Input:
	grid_size: [2] tf.int, height and width of the grid
	angle_range: [2,3] tf.float32, min and max euler angles

	Output:
	angles: [prod(grid_size), 3] tf.float32, uniformly spaced angles
	"""

	def linspace(n):
		if n == 1:
			return tf.constant([0.5])
		return tf.linspace(0., 1., n)
	gh = grid_size[-2]; gw = grid_size[-1]
	y, x = tf.meshgrid(linspace(gw), linspace(gh))
	angles = tf.stack([x, y, 0.5+tf.zeros_like(x)], -1)
	angles = tf.reshape(angles, [gw*gh, 3])
	angles = angles * (angle_range[1]-angle_range[0]) + angle_range[0]
	return angles

def stack_images(im, grid_size=None):
	"""
	It stacks a batch of images into a big displayable image.

	Input:
	im: [B,H,W,3] or [B,H,W] tf.float32, batch of images
	grid_size: [2] tf.float32, shape of the output grid,
		where prod(grid_size) = B

	Output:
	ims: [H*gh, W*gw, C] or [H*gh, W*gw] tf.float32, stacked images
	"""
	
	s = _shape(im)
	B, H, W = s[:3]
	if grid_size is None:
		grid_size = [B, 1]
	gh = grid_size[-2]; gw = grid_size[-1]
	ims = tf.reshape(im, [gh, gw, H, W, -1])
	ims = tf.transpose(ims, [0,2,1,3,4])
	ims = tf.reshape(ims, [H*gh, W*gw, *s[3:]])
	return ims

def alpha_matte(fg, bcg, apply_alpha=False):
	"""
	I does alpha matting on images.
	T foreground fg and the background is bcg.

	Input:
	fg: [B, H, W, C+1] tf.float32, foreground image
		C+1 channel array, where fg[:,:,:,0:C] is the image
		and fg[:,:,:,C] is the alpha channel.
	bcg: [B, H, W, C+1] tf.float32, backgroung image
		The format is the same as for fg.
	apply_alpha: bool, indicates whether to multiply the images with
		alpha channel before matting

	The first dimension is the batch, which is broadcasted.

	Output:
	im: [B, H, W, C+1] tf.float32, the composed image
	"""
	
	if apply_alpha:
		fg = tf.concat([fg[...,:-1]*fg[...,-1:], fg[...,-1:]], -1)
		bcg = tf.concat([bcg[...,:-1]*bcg[...,-1:], bcg[...,-1:]], -1)
	
	im = fg + (1.-fg[:,:,:,-1:]) * bcg
	return im

def filter2D(x, f, data_format='NHWC'):
	# x: [B,H,W,C]
	# f: [kh,kw,1,1]
	kh, kw = _shape(f)
	C = _shape(x)[-1]
	ph = (kh-1) // 2; pw = (kw-1) // 2
	x = tf.pad(x, [[0,0],[ph,kh-1-ph],[pw,kw-1-pw],[0,0]], 'SYMMETRIC')
	f = tf.tile(f[:,:,tf.newaxis,tf.newaxis], [1,1,C,1])
	y = tf.nn.depthwise_conv2d(x, f, [1,1,1,1], padding='VALID', data_format=data_format)
	return y

def gaussian_kernel(sigma, ksize):
	x = tf.cast(tf.range(2*ksize+1)-ksize, tf.float32)
	w = tf.exp( -0.5 * (x**2) / sigma**2)
	w /= tf.reduce_sum(w)
	return w

def gaussian_blur(x, sigma, data_format='NHWC'):
	"""
	It applies Gaussian blur on x with sigma.

	Input:
	x: [B, H, W, C] tf.float32, image
	sigma: [] tf.float32, parameter of the Gaussian.

	Output:
	y: [B, H, W, C] tf.float32, blurred image
	"""
	
	ksize = tf.cast(tf.math.ceil(sigma*3.), tf.int32)
	f = gaussian_kernel(sigma, ksize)
	y = filter2D(filter2D(x, f[:,tf.newaxis]), f[tf.newaxis,:])
	return y

