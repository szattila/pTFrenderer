"""
Copyright (C) 2020 Attila Szabo
All rights reserved.
This file is part of the pTFrenderer library and is made available under
the terms of the BSD license (see the LICENSE file).
"""

import math
import tensorflow as tf

from . import utils
from . import camera
from . import mesh

_broadcast = utils._broadcast

def render(xyz, attr, vertuv, triangles, K, P, imsize,
    bcg=0., **kwargs):
    """
    It renders an image using the UV mapping technique.

    Input:
    xyz: [B, H1, W1, 3] tf.float32, surface map
    attr: [B, H2, W2, C] tf.float32, attribute map, its resolution
        can be different from surface map's.
    vertuv: [B, N, 2] tf.float32, UV coordinates, whrere
        [-1,-1] and [+1,+1] are the coordinates of the top left
        and bottom right corner of the centered rectangle whose
        height is aligned with the height of the sampled image.
        See "utils.sample2D_bilinear" for more details.
    triangles: [M, 3] tf.int, list of vertex indices
    K: [B,3,3] tf.float32, intrisic camera matrices
        See "mesh.render" more details.
    P: [B,3,4] tf.float32, extrinsic camera matrices
    imsize: [2] tf.int, the size of image in pixels
        imsize = [H, W], where H is the image height and W is the width
    bcg: [B, H, W, C+1] tf.float32, optional background image.
        Its size has to match the rendered output image or should be able
        to broadcast to it.
        It is required to have an alpha channel, which is bcg[...,-1].
    kwargs: keyword arguments passed to the "mesh.render" function.

    The first dimension is the batch, which is broadcasted for the
    following inputs: xyz, attr, vertuv, K, P.

    Output:
    image: [B,H,W,C+1] tf.float32, the first C channels are the
        rendered attributes and the last channel is the alpha
    """

    xyz, attr, vertuv, K, P = _broadcast(xyz, attr, vertuv, K, P)
    vertxyz = utils.sample2D_bilinear(xyz, vertuv)
    def uvmap(imuva):
        imattr = utils.sample2D_bilinear(attr, imuva[...,:2])
        imattr = tf.concat([imattr, imuva[...,2:]], -1)
        return imattr
    im_crisp, im_margin = mesh.render_(
        vertxyz, vertuv, triangles, K, P, imsize, **kwargs)
    image = utils.alpha_matte(uvmap(im_margin), uvmap(im_crisp), apply_alpha=True)
    image = utils.alpha_matte(image, bcg)
    return image

def normals(xyz):
    """
    It computes normals for the surface map xyz.

    Input:
    xyz: [B, H1, W1, 3] tf.float32, surface map

    Output:
    n: [B, H1, W1, 3] tf.float32, surface normals
    """

    f = tf.constant([-1., 0., 1.])
    dx = utils.filter2D(xyz, f[tf.newaxis,:])
    dy = utils.filter2D(xyz, f[:,tf.newaxis])
    n = tf.math.l2_normalize(tf.linalg.cross(dy, dx), -1)
    return n

def sheet_init(imsize):
    """
    It creates an 3 channel image tha that represents
    the 3D coordinates of a surface map of a sheet.
    Th sheet is stretces between -1 and +1 for both
    x and y coordinates and z is 0.

    Input:
    imsize: [2] tf.int, the image size

    Output:
    xyz: [1, imsize[-2], imsize[-1], 3] tf.float32, the surface map
    """

    H = imsize[-2]; W = imsize[-1]
    # vertices
    x, y = tf.meshgrid(tf.linspace(-1., 1., W),
        tf.linspace(-1., 1., H))
    xyz = tf.stack([x, y, tf.zeros_like(x)], -1)[tf.newaxis,:,:,:]
    return xyz

def sheet(meshsize, direction=0, maskradius=None):
    """
    It creates a regular triangular grid mesh.
    The vertices denote UV coordinates, whic can be used to sample
    texture images and surface maps.

    Input:
    meshsize: [2], tf.int, the resolution of the grid

    Output:
    vertuv: [B, N, 2] tf.float32, UV coordinates, whrere
        [-1,-1] and [+1,+1] are the coordinates of the top left
        and bottom right corner of the centered rectangle whose
        height is aligned with the height of the sampled image.
        See "utils.sample2D_bilinear" for more details.
    triangles: [M, 3] tf.int, list of vertex indices
    """

    H = meshsize[-2]; W = meshsize[-1]

    u, v = tf.meshgrid(tf.linspace(-1., 1., W), tf.linspace(-1., 1., H))
    vertuv = tf.reshape(tf.stack([u, v], -1), [1, H*W, 2])

    def triangles_sheet():
        idx = tf.reshape(tf.range(H*W), [H,W])
        v00 = tf.reshape(idx[0:H-1,0:W-1], [-1])
        v01 = tf.reshape(idx[0:H-1,1:W], [-1])
        v10 = tf.reshape(idx[1:H,0:W-1], [-1])
        v11 = tf.reshape(idx[1:H,1:W], [-1])
        if direction==0:
            t00 = tf.stack([v00, v10, v01], -1)
            t11 = tf.stack([v11, v01, v10], -1)
            t = tf.concat([t00, t11], -2)
        if direction==1:
            t10 = tf.stack([v10, v11, v00], -1)
            t01 = tf.stack([v01, v00, v11], -1)
            t = tf.concat([t01, t10], -2)
        if maskradius is not None:
            mask = tf.reduce_sum(vertuv[0]**2, -1) <= maskradius**2
            vinside = tf.gather(mask, t)
            tinside = tf.reduce_all(vinside, -1)
            t = tf.boolean_mask(t, tinside)
        return t

    return vertuv, triangles_sheet()

def sphere_init(imsize):
    """
    It creates an 3 channel image tha that represents
    the 3D coordinates of a surface map of a unit sphere.

    Input:
    imsize: [2] tf.int, the image size

    Output:
    xyz: [1, imsize[-2], imsize[-1], 3] tf.float32, the surface map
    """

    H = imsize[-2]; W = imsize[-1]
    # vertices
    u, v = tf.meshgrid(tf.linspace(-1., 1., W),
        tf.linspace(-1., 1., H))
    r = tf.sqrt(u**2 + v**2)
    angle = r * math.pi
    x = tf.math.divide_no_nan(u * tf.sin(angle), r)
    y = tf.math.divide_no_nan(v * tf.sin(angle), r)
    z = -tf.cos(angle)
    xyz = tf.stack([x, y, z], -1)[tf.newaxis,:,:,:]
    return xyz

def sphere(meshsize):
    """
    It creates a triangle mesh with a sphere topology.
    The vertices denote UV coordinates, whic can be used to sample
    texture images and surface maps.

    Input:
    meshsize: [2], tf.int, the resolution of the grid

    Output:
    vertuv: [B, N, 2] tf.float32, UV coordinates, whrere
        [-1,-1] and [+1,+1] are the coordinates of the top left
        and bottom right corner of the centered rectangle whose
        height is aligned with the height of the sampled image.
        See "utils.sample2D_bilinear" for more details.
    triangles: [M, 3] tf.int, list of vertex indices
    """

    def odd(n):
        return 1+2*(n//2)
    H = odd(meshsize[-2]); W = odd(meshsize[-1])

    def triangles_shphere():
        idx = tf.reshape(tf.range(H*W), [H, W])
        # vertices [H-1, W-1]
        v00 = idx[0:H-1,0:W-1]
        v01 = idx[0:H-1,1:W]
        v10 = idx[1:H,0:W-1]
        v11 = idx[1:H,1:W]
        # triangles [H-1, W-1, 3]
        t00 = tf.stack([v00, v10, v01], -1)
        t01 = tf.stack([v01, v00, v11], -1)
        t10 = tf.stack([v10, v11, v00], -1)
        t11 = tf.stack([v11, v01, v10], -1)
        # sheets [h-1, w-1, 2, 3]
        s0011 = tf.stack([t00, t11], -2)
        s0110 = tf.stack([t01, t10], -2)
        # combine
        s00 = s0011[0:(H-1)//2, 0:(W-1)//2]
        s01 = s0110[0:(H-1)//2, (W-1)//2:]
        s10 = s0110[(H-1)//2:, 0:(W-1)//2]
        s11 = s0011[(H-1)//2:, (W-1)//2:]
        def r(s): return tf.reshape(s, [-1,3])
        triangles = tf.concat([r(s00), r(s01), r(s10), r(s11)], axis=-2)
        return triangles
    
    def vert_3D():
        x, y = tf.meshgrid(tf.linspace(-1., 1., W), tf.linspace(-1., 1., H))
        z = tf.abs(x)+tf.abs(y)-1
        mask = tf.cast(z<=0, tf.float32)
        x2 = mask*x + (1-mask)*(1-tf.abs(y))*tf.sign(x)
        y2 = mask*y + (1-mask)*(1-tf.abs(x))*tf.sign(y)
        xyz = tf.reshape(tf.stack([x2, y2, z],-1), [1,H*W,3])
        xyz = tf.math.l2_normalize(xyz, -1)
        return xyz

    def vert_to_2D(xyz):
        xyz = tf.math.l2_normalize(xyz, -1)
        x = xyz[:,:,0:1]; y = xyz[:,:,1:2]
        cosa = -xyz[:,:,2:3]
        sina = tf.sqrt(x**2 + y**2)
        a = tf.atan2(sina, cosa)
        uv = tf.math.divide_no_nan(tf.concat([x, y], -1) * a, sina * math.pi)
        uvback = tf.constant([[[0., -1.]]])
        isback = tf.logical_and(tf.abs(sina) < 1e-6, cosa < 0)
        uv = tf.compat.v2.where(isback, uvback, uv)
        return uv
    vertuv = vert_to_2D(vert_3D())
    triangles = triangles_shphere()
    return vertuv, triangles

