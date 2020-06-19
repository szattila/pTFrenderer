"""
Copyright (C) 2020 Attila Szabo
All rights reserved.
This file is part of the pTFrenderer library and is made available under
the terms of the BSD license (see the LICENSE file).
"""

import tensorflow as tf

from . import utils
from . import camera

_i64 = utils._i64
_f32 = utils._f32
_shape = utils._shape
_broadcast = utils._broadcast
_eps = 1e-3
_div = tf.math.divide_no_nan
_where = tf.compat.v2.where


def face_attr(va, triangles):
	"""
	It gathers vertex attributes va for the triangle faces.

	Input:
	va: [B, N, C] tf.float32, vertex attributes of N vertices of C channels
	triangles: [M, 3] tf.int, list of vertex indices of M triangles

	Output:
	fa: [B, M, 3, C] tf.float32, attributes of face vertices
	"""

	B = _shape(va)[0]
	M = _shape(triangles)[-2]
	triangles = _i64(triangles)

	t = tf.tile(triangles[tf.newaxis,:,:], [B,1,1])
	fa = tf.gather(va, t, batch_dims=1)
	return fa

def face_normals(xyz, triangles):
	"""
	It computes normals for triangle faces.

	Input:
	xyz: [B, N, 3] tf.float32, 3D vertex locations
	triangles: [M, 3] tf.int, list of vertex indices of M triangles

	Output:
	fn [B, M, 3] tf.float32, face normals
	"""

	abc_xyz = face_attr(xyz, triangles)

	bc_xyz = abc_xyz[:,:,1:3] - abc_xyz[:,:,0:1]
	fn = tf.linalg.cross(bc_xyz[:,:,0], bc_xyz[:,:,1])
	fn = tf.math.l2_normalize(fn, -1)
	return fn

def vert_normals(xyz, triangles):
	"""
	It computes normals for vertices xyz by averaging face normals on
	triangles that each vertex belongs to.

	Input:
	xyz: [B, N, 3] tf.float32, 3D vertex locations
	triangles: [M, 3] tf.int, list of vertex indices of M triangles

	Output:
	vn: [B, N, 3] tf.float32, vertex normals
	"""

	B, N, _ = _shape(xyz)
	M = _shape(triangles)[-2]
	triangles = _i64(triangles)
	
	fn = face_normals(xyz, triangles)
	bfn = tf.reshape(tf.tile(fn, [1,1,3]), [B*M*3, 3])
	bt = tf.reshape(
		triangles[tf.newaxis,:,:] + _i64(tf.range(B)[:,tf.newaxis,tf.newaxis] * N),
		[B*M*3])
	vn = tf.reshape(tf.math.unsorted_segment_sum(bfn, bt, B*N), [B,N,3])
	vn = tf.math.l2_normalize(vn, -1)
	return vn

def render(xyz, attr, triangles, K, P, imsize, correct=True, **kwargs):
	"""
	It renders a batch of images.

	Input:
	xyz: [B, N, 3] tf.float32, 3D vertex locations in World frame, N >= 1
	attr: [B, N, C] tf.float32, 3D vertex attributes
	triangles: [M, 3] tf.int, list of vertex indices
	K: [B,3,3] tf.float32, intrisic camera matrices
		K*xyz gives the 2D homogenous coordinates of vertices in clip-space.
		The top row of pixels have coordinates [_,-1] and
		the bottom row have [_,+1] in the rendered image.
		Thus, for rectangular images the clip-space region [-1,+1]x[-1,+1]
		is rendered, so [-1,-1] is the center of the top-left pixel and
		[+1,+1] is the center of the bottom-right pixel.
	P: [B,3,4] tf.float32, extrinsic camera matrices
	imsize: [2] tf.int, the size of image in pixels
		imsize = [H, W], where H is the image height and W is the width
	correct: bool, if True it applies perspecitve correction to the barycentric
		coordinates.
	kwargs: keyword arguments passed to the "rasterize" function.

	The first dimension is the batch, which is broadcasted for the
	following inputs: xyz, attr, K, P.

	Output:
	image: [B,H,W,C+1] tf.float32, the first C channels are the
		rendered attributes and the last channel is the alpha
	"""

	image_margin, image_crisp = render_(
		xyz, attr, triangles, K, P, imsize, correct, **kwargs)
	image = utils.alpha_matte(image_margin, image_crisp, apply_alpha=True)
	return image

def render_(xyz, attr, triangles, K, P, imsize, correct=True, **kwargs):
	xyz, attr, K, P = _broadcast(xyz, attr, K, P)
	triangles = _i64(triangles)

	H = imsize[-2]; W = imsize[-1]
	B, N, C = _shape(attr)

	xyz = camera.transform(P, xyz)
	uv = camera.project(K, xyz) # uv is in clip space
	uvz = tf.concat([(_f32(H)-1)*(uv[:,:,0:1]+1)/2. + (_f32(W)-_f32(H))/2.,
		             (_f32(H)-1)*(uv[:,:,1:2]+1)/2.,
		             xyz[:,:,2:3]], -1) # uv is in pixel space

	btuv, bary, _, a = rasterize(uvz, triangles, imsize, **kwargs)

	vidx = tf.gather(triangles, btuv[:,1]) + btuv[:,0:1] * _i64(N)
	zattr = tf.reshape(tf.concat([xyz[:,:,2:], attr], -1), [B*N,1+C])
	face_zattr = tf.gather(zattr, vidx) # [L, 3, 3+C]

	if correct:
		z = 1. / interp_bary(1. / face_zattr[:,:,:1], bary)
		bary *= z / face_zattr[:,:,0]

	pixel_attr = interp_bary(face_zattr[:,:,1:], bary)
	pixel_attr_a = tf.concat([pixel_attr, a],-1)

	pidx = btuv[:,0]*_i64(H)*_i64(W) + btuv[:,3]*_i64(W) + btuv[:,2]
	def draw(x, valid):
		x_v = tf.concat([x*valid, valid], -1)
		imx_n = tf.math.unsorted_segment_sum(x_v, pidx, B*H*W)
		imx = tf.reshape(_div(imx_n[:,:-1], imx_n[:,-1:]), [B,H,W,-1])
		return imx

	a_crisp = _f32(tf.math.equal(a, 1.))
	image_margin = draw(pixel_attr_a, 1.-a_crisp)
	image_crisp = draw(pixel_attr_a, a_crisp)
	return image_margin, image_crisp

def rasterize(uvz, triangles, imsize,
	margin=1., znear=1e-3, zfar=1e3):
	"""
	It rasterises triangles.

	Input:
	uvz: [B, N, 3] tf.float32, 2D vertex locations (in pixels) and depth value
		B is the batch size and N >= 1 is the number of vertices.
		[0., 0.] is the center of the top-left pixel and
		[W-1, H-1] is center of the bottom-right pixel.
	triangles: [M, 3] tf.int, list of vertex indices of M triangles
		The same triangle mesh is applied for the whole batch.
		Trinagles that have a vertex closer to znear will be ignored.
		We apply culling, back-facing triangles are not drawn.
	imsize: [2] tf.int, the size of image in pixels
		imsize = [H, W], where H is the image height and W is the width
	margin: tf.float32, radius of matting region in pixels
		margin >= 0.
		when margin == 0., there is no matting
	znear, zfar: tf.float32, the valid z range, where 0. < znear < zfar

	Output:
	Pixel data for L pixels accross the full batch.
	tbvu: [L, 4] tf.int, triangle ID, batch ID, and pixel positions v and u
	bary: [L, 3] tf.float32, barycentric coordinates
	z: [L, 1] tf.float32, depth values
	alpha: [L, 1] tf.float32, translucency value
	"""

	margin = tf.maximum(margin, 0.)
	triangles = _i64(triangles)

	H = imsize[-2]; W = imsize[-1]
	B, N, _ = _shape(uvz)
	M, _ = _shape(triangles)

	def triangle_bounds():
		abc = face_attr(uvz, triangles)
		abc = tf.reshape(abc, [B*M,3,3])
		def bound(x):
			xb = tf.clip_by_value(_i64(x), 0, [[_i64(W), _i64(H)]])
			return xb
		uvmin = bound(tf.math.ceil(tf.reduce_min(abc[:,:,0:2]-margin, -2)))
		uvmax = bound(tf.math.floor(tf.reduce_max(abc[:,:,0:2]+margin, -2))+1)
		# ignore triangles too close
		isclose = tf.reduce_min(abc[:,:,2:3], -2) < znear
		uvmax = _where(isclose, uvmin, uvmax)
		# culling
		isbackward = detA(abc[:,0,:], abc[:,1,:], abc[:,2,:]) > -_eps
		uvmax = _where(isbackward[:,tf.newaxis], uvmin, uvmax)
		return abc, uvmin, uvmax

	def segment_fun():
		abc, uvmin, uvmax = triangle_bounds()
		nuv = uvmax-uvmin; nu = nuv[:,0:1]; nv = nuv[:,1:2]
		npix = nu*nv

		tb, pidx, [uvmin, nu] = segment(npix, [uvmin, nu])
		abc = tf.gather(abc, tb)

		b = tb // _i64(M)
		t = tb % _i64(M)
		u = uvmin[:,0] + pidx % nu[:,0]
		v = uvmin[:,1] + pidx // nu[:,0]
		btuv = tf.stack([b, t, u, v], -1)
		return btuv, abc

	def map_fun(btuv, abc):
		abc_uv = abc[:,:,:2] - _f32(btuv[:,tf.newaxis,2:4])

		uv, d = closest_point_on_triangle_to_origin(abc_uv)
		bary, _ = barycentric(abc_uv-uv[:,tf.newaxis,:])
		z = interp_bary(abc[:,:,2:], bary)
		slope = 0.5
		z *= (1. + slope * d / tf.maximum(margin, _eps))
		alpha = tf.clip_by_value(1. - d / tf.maximum(margin, _eps), 0., 1.)
		
		return [bary, z, alpha]

	def reduce_fun(btuv, v2):
		bary, z, alpha = v2
		pidx = btuv[:,0]*_i64(H)*_i64(W) + btuv[:,3]*_i64(W) + btuv[:,2]
		def my_min(x):
			xmin = tf.gather(tf.math.unsorted_segment_min(x, pidx, B*H*W), pidx)
			return xmin

		# check depth
		valid = tf.logical_and(z < zfar, znear <= z)

		# crisp
		valid_crisp = tf.logical_and(valid, tf.math.equal(alpha, 1.))
		z_crisp = _where(valid_crisp, z, zfar)
		z_min = my_min(z_crisp)
		valid_crisp = tf.logical_and(valid_crisp, tf.math.equal(z_crisp, z_min))

		# margin
		valid_margin = tf.logical_and(valid, alpha<1.)
		valid_margin = tf.logical_and(valid_margin, 0.<alpha)
		valid_margin = tf.logical_and(valid_margin, z<z_min)
		z_margin = _where(valid_margin, z, zfar)
		valid_margin = tf.logical_and(valid_margin, tf.math.equal(z_margin, my_min(z_margin)))

		valid = tf.logical_or(valid_crisp, valid_margin)
		valid = tf.reshape(valid, [-1])
		return valid

	btuv, [bary, z, alpha]= map_reduce(segment_fun, map_fun, reduce_fun)
	return btuv, bary, z, alpha

def map_reduce(segment_fun, map_fun, reduce_fun):
	keys, v1 = segment_fun()
	v2 = map_fun(keys, v1)
	valid = reduce_fun(keys, v2)
	keys = tf.boolean_mask(keys, valid)
	v2 = [tf.boolean_mask(x, valid) for x in v2]
	return keys, v2

def multi_gather(indices, params):
	# The function does this:
	# outs = [tf.gather(p, indices) for p in params]

	n = len(params)
	params = [tf.convert_to_tensor(params[i]) for i in range(n)]
	shapes = [_shape(params[i]) for i in range(n)]
	sizes = [tf.size(params[i][0]) for i in range(n)]
	slice_bounds = tf.pad(tf.cumsum(sizes), [[1, 0]])

	cat_params = tf.concat(
		[tf.reshape(params[i], [-1,sizes[i]]) for i in range(n)],
		-1)
	cat_outs = tf.gather(cat_params, indices)
	
	outs = [cat_outs[:, slice_bounds[i]:slice_bounds[i+1]] for i in range(n)]
	outs = [tf.reshape(outs[i], [-1, *shapes[i][1:]]) for i in range(n)]
	return outs

def segment(lengths, params):
	"""
	It creates segment ID and index within the segments.
	It also gathers the parametes according to the segment ID.

	The function was inspired by the answer of the user "jdehesa" on stackoverflow.
	https://stackoverflow.com/questions/54790000
	"""
	
	lengths = lengths[:,0]
	S = tf.cumsum(lengths)
	index = tf.range(S[-1], dtype=tf.int64)
	sid = tf.searchsorted(S, index, side='right', out_type=tf.dtypes.int64)
	outs = multi_gather(sid, [lengths-S, *params])
	index += outs[0] # offset
	outs = outs[1:]
	return sid, index, outs

def detA(a, b, c):
	u = 0; v = 1
	d = a[:,u]*(b[:,v]-c[:,v]) \
	  - b[:,u]*(a[:,v]-c[:,v]) \
	  + c[:,u]*(a[:,v]-b[:,v])
	return d

def barycentric(abc):
	a = abc[:,0,:]; b = abc[:,1,:]; c = abc[:,2,:]
	_0 = tf.zeros_like(a)
	barya = detA(_0,b,c)
	baryb = detA(a,_0,c)
	baryc = detA(a,b,_0)
	bary = tf.stack([barya, baryb, baryc], -1)
	s = tf.reduce_sum(bary, -1)
	bary = _div(bary, s[:,tf.newaxis])
	isinside = tf.logical_and(tf.reduce_sum(bary, -1) > 0.5,
		tf.reduce_all(bary >= 0., -1))[:,tf.newaxis]
	return bary, isinside

def interp_bary(x, bary):
	# x: [L, 3, C]
	# bary: [L, 3]
	# y: [L, C]
	y = tf.reduce_sum(x * bary[:,:,tf.newaxis], -2)
	return y

def closest_point_on_triangle_to_origin(abc):
	a = abc[:,0,:]; b = abc[:,1,:]; c = abc[:,2,:]
	def closest_on_line_segment(p, q):
		n = p-q
		l = tf.norm(n, axis=-1, keepdims=True)
		n = _div(n, l)
		uv = q - n * tf.clip_by_value(
			tf.reduce_sum(n*q, -1, keepdims=True), -l, 0.)
		d = tf.norm(uv, axis=-1, keepdims=True)
		isinside = tf.logical_or(
			n[:,1:2]*q[:,0:1] - n[:,0:1]*q[:,1:2] >= 0.,
			d == 0.)
		return uv, d, isinside
	def update_uv(uv, d, isinside, p, q):
		uv2, d2, isinside2 = closest_on_line_segment(p,q)
		closer = d2 < d
		uv = _where(closer, uv2, uv)
		d = _where(closer, d2, d)
		isinside = tf.logical_and(isinside, isinside2)
		return uv, d, isinside
	uv, d, isinside = closest_on_line_segment(a,b)
	uv, d, isinside = update_uv(uv, d, isinside, b, c)
	uv, d, isinside = update_uv(uv, d, isinside, c, a)
	d = _where(isinside, 0., d)
	uv = _where(isinside, 0., uv)
	return uv, d

