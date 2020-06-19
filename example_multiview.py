
# %cd '/content/drive/My Drive/code/renderer'
# !ln -sf /opt/bin/nvidia-smi /usr/bin/nvidia-smi
# import subprocess
# print(subprocess.getoutput('nvidia-smi'))

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.compat.v1.enable_eager_execution()

import argparse
import matplotlib.pyplot as plt
from PIL import Image

from ptfrenderer import utils
from ptfrenderer import camera
from ptfrenderer import uv
_shape = utils._shape
_broadcast = utils._broadcast


# inputs
parser = argparse.ArgumentParser()
parser.add_argument('-save_dir', default=None)
parser.add_argument('-res', default=128, type=int)
parser.add_argument('-meshres', default=128, type=int)
parser.add_argument('-nviews', default=1000, type=int)
parser.add_argument('-margin', default=1., type=float)
parser.add_argument('-blur_mesh', default=2.0, type=float)
parser.add_argument('-batch_size', default=50, type=int)
parser.add_argument('-epochs', default=10, type=int)
parser.add_argument('-lrate', default=0.01, type=float)
args = parser.parse_args()

# model
rgbres = args.res
meshres = args.meshres
bcgres = args.res*3

# rendering
imres = args.res
angle_range = [[-60., -90., 0.], [60., 90., 0.]]
fow = 10.*np.pi/180.
margin = args.margin

# visualize result
grid_size = [5, 7]


def load_im(path):
	im = np.asarray(Image.open(path))
	im = im.astype(np.float32) / 255.
	return im

def save_im(im, path):
	im = np.array(tf.cast(tf.clip_by_value(im, 0., 1.)*255., tf.uint8))
	pilim = Image.fromarray(im)
	pilim.save(path)
	return

class Scene(tf.keras.Model):
	def __init__(self, xyz, rgb, bcg, meshres, imres, angle_range, fow, margin, blur_mesh=2.0, isgt=False):
		super(Scene, self).__init__()
		self.xyz = tf.Variable(xyz)
		self.rgb = tf.Variable(rgb)
		self.bcg = tf.Variable(bcg)
		self.vertuv, self.triangles = uv.sphere([meshres, meshres])

		self.imsize = tf.constant([imres, imres])
		self.angle_range = tf.constant(angle_range) * np.pi/180.
		self.fow = tf.constant(fow)
		self.margin = tf.constant(margin)
		self.blur_mesh = blur_mesh

		self.isgt = isgt

	def get_xyz(self):
		if self.isgt:
			return self.xyz
		xyz = utils.gaussian_blur(self.xyz, self.blur_mesh)
		# blurring is necessary because of the uv sampling and stability of training
		return xyz

	def get_cam_params(self, angles):
		K, T = camera.look_at_origin(self.fow)
		angles_cam = angles * (self.angle_range[1]-self.angle_range[0]) + self.angle_range[0]
		R = camera.euler(angles_cam)
		R, T = _broadcast(R, T)
		P = tf.matmul(camera.hom(T,'P'), camera.hom(R,'R'))[:,:3,:]
		return K, P

	def get_bcg_cropped(self, angles):
		# simulates bcg at infinity somewhat
		bcg, angles = _broadcast(self.bcg, angles)
		B, H, W, _ = _shape(bcg)
		rh = tf.cast(imres, tf.float32) / tf.cast(H, tf.float32)
		rw = tf.cast(imres, tf.float32) / tf.cast(W, tf.float32)
		y1 = (1.-rh)*angles[:,0]
		x1 = (1.-rw)*angles[:,1]
		boxes = tf.stack([y1, x1, y1+rh, x1+rw], -1)
		bcg = tf.image.crop_and_resize(bcg, boxes, tf.range(B), self.imsize)
		bcg = tf.pad(bcg, [[0,0],[0,0],[0,0],[0,1]], constant_values=1.)
		return bcg

	def call(self, angles, isnorm=False):
		# renders the scene form the specified angles
		xyz = self.get_xyz()
		def l2_normalize(ima):
			im = tf.math.l2_normalize(ima[...,:-1], -1)
			a = ima[...,-1:]
			ima = tf.concat([(im+1.)/2., a], -1)
			return ima
		if isnorm:
			attr = uv.normals(xyz)
			bcg = 1.
			margin = 0.
			postfun = l2_normalize
		else:
			attr = self.rgb
			bcg = self.get_bcg_cropped(angles)
			margin = self.margin
			postfun = lambda x: x
		K, P = self.get_cam_params(angles)
		ima = uv.render(xyz, attr, self.vertuv, self.triangles,
			K, P, self.imsize, bcg=bcg, margin=margin)
		ima = postfun(ima)
		return ima[...,:3] # ignore alpha

# ground truth scene
def deformed_sphere():
	xyz = uv.sphere_init([rgbres, rgbres])
	x = xyz[:,:,:,0]; y = -xyz[:,:,:,1]; z = -xyz[:,:,:,2]
	xt = tf.minimum(tf.maximum(2*x,-2),2) + 0.5*tf.sin(x) + 0.2*tf.cos(5*y) + 0.5*z*z
	yt = 2*y + 0.5*tf.sin(x+2*y) + 0.2*tf.cos(5*z) - 0.3*x*z
	zt = 2*z + 0.5*tf.sin(z-3*x) + 0.2*tf.cos(15*y+6*z) + 0.3*y*x
	xyz = 0.25 * tf.stack([xt,-yt,-zt],axis=-1)
	return xyz
xyz = deformed_sphere()
rgb = load_im("assets/fish.jpg")
rgb = tf.image.resize(rgb[tf.newaxis,...,:3], [rgbres, rgbres])
bcg = load_im("assets/lake.jpg")
bcg = tf.image.resize(bcg[tf.newaxis,...,:3], [bcgres, bcgres])
ground_truth = Scene(xyz, rgb, bcg, meshres, imres, angle_range, fow, margin, isgt=True)

# ground truth data
print("Generate training data ...")
angles_train = tf.random.uniform([args.nviews, 3])
# ima_train = ground_truth(angles_train) # this might not fit into memory
ima_train = tf.concat([ground_truth(angles_train[i:i+1]) for i in range(args.nviews)], 0)
print("done.")

# estimated scene
xyz = 0.5 * uv.sphere_init([rgbres, rgbres])
rgb = 0.5 + tf.zeros([1, rgbres, rgbres, 3])
bcg = 0.5 + tf.zeros([1, bcgres, bcgres, 3])
model = Scene(xyz, rgb, bcg, meshres, imres, angle_range, fow, margin, blur_mesh=args.blur_mesh)

def loss_l2(ypred, y):
	return tf.reduce_mean((ypred-y)**2, axis=[1,2,3])

print("Train model ...")
print("image resolution: {} x {}".format(imres, imres))
print("number of vertices: {}".format(_shape(model.vertuv)[-2]))
print("number of triangles: {}".format(_shape(model.triangles)[-2]))
model.compile(
	optimizer=tf.keras.optimizers.Adam(learning_rate=args.lrate, beta_1=0.9, beta_2=0.999),
	loss=loss_l2)
model.fit(x=angles_train, y=ima_train, batch_size=args.batch_size, epochs=args.epochs)
print("done.")

print("Evaluate model ...")
angles_test = utils.grid_angles(grid_size)
im_test = tf.concat([ground_truth(angles_test[i:i+1]) for i in range(tf.reduce_prod(grid_size))], 0)
model.evaluate(angles_test, im_test, verbose=2)
print("done.")

print("Visualize results ...")
im_test = utils.stack_images(im_test, grid_size)
norm_test = tf.concat([ground_truth(angles_test[i:i+1], isnorm=True) for i in range(tf.reduce_prod(grid_size))], 0)
norm_test = utils.stack_images(norm_test, grid_size)

im_pred = tf.concat([model(angles_test[i:i+1]) for i in range(tf.reduce_prod(grid_size))], 0)
im_pred = utils.stack_images(im_pred, grid_size)
norm_pred = tf.concat([model(angles_test[i:i+1], isnorm=True) for i in range(tf.reduce_prod(grid_size))], 0)
norm_pred = utils.stack_images(norm_pred, grid_size)

if args.save_dir is None:
	plt.figure(1); plt.imshow(im_pred, interpolation='nearest')
	plt.figure(2); plt.imshow(norm_pred, interpolation='nearest')
	plt.figure(3); plt.imshow(im_test, interpolation='nearest')
	plt.figure(4); plt.imshow(norm_test, interpolation='nearest')
	plt.show()
else:
	save_dir = os.path.join(args.save_dir, "imres={} margin={}".format(imres, margin))
	os.makedirs(save_dir, exist_ok=True)
	save_im(im_pred, os.path.join(save_dir, 'im_pred.png'))
	save_im(norm_pred, os.path.join(save_dir, 'norm_pred.png'))
	save_im(im_test, os.path.join(save_dir, 'im_test.png'))
	save_im(norm_test, os.path.join(save_dir, 'norm_test.png'))

print("done.")
