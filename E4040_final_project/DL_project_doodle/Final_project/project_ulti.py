'''
We define some basic tool in project_ulti.py
These tool include: kmeans, load_mask_labels, preprocess_image, deprocess_image, batch_flatten, gram_matrix
Some of these tool are reference from Github: https://github.com/jcjohnson/neural-style
https://github.com/DmitryUlyanov/fast-neural-doodle
Many thanks to the authors.
'''

import os
import time
from scipy.misc import imread, imsave, imresize, fromimage, toimage, imfilter
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave, imread, imresize
import theano.tensor.nnet
import theano.tensor.nnet.neighbours
from keras import backend as K
import theano
import theano.tensor as T
from keras.applications import vgg16
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Input, AveragePooling2D
from keras.models import Model
import numpy as np

img_nrows = 400
img_ncols = 400
nb_labels = 4
nb_colors = 3  
style_weight = 1.
region_style_weight=0.4
root_dir = os.path.abspath('.')

ref_image_path = os.path.join(root_dir, 'style.png')
style_mask_path = os.path.join(root_dir,'style_mask.png')
target_mask_path = os.path.join(root_dir,'target_mask.png')
def kmeans(xs, k):
    #assert xs.ndim == 2
    try:
        from sklearn.cluster import k_means
        _, labels, _ = k_means(xs.astype("float64"), k)
    except ImportError:
        from scipy.cluster.vq import kmeans2
        _, labels = kmeans2(xs, k, missing='raise')
    return labels

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    return img

def deprocess_image(x):
    x = x.reshape((3, img_nrows, img_ncols))
    x = x.transpose((1, 2, 0))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def load_mask_labels():
    '''Load both target and style masks.
    A mask image (nr x nc) with m labels/colors will be loaded
    as a 4D boolean tensor: (1, m, nr, nc) for 'th' or (1, nr, nc, m) for 'tf'
    '''
    target_mask_img = load_img(target_mask_path,
                               target_size=(img_nrows, img_ncols))
    target_mask_img = img_to_array(target_mask_img)
    style_mask_img = load_img(style_mask_path,
                              target_size=(img_nrows, img_ncols))
    style_mask_img = img_to_array(style_mask_img)

    mask_vecs = np.vstack([style_mask_img.reshape((3, -1)).T,
                           target_mask_img.reshape((3, -1)).T])


    labels = kmeans(mask_vecs, nb_labels)
    style_mask_label = labels[:img_nrows *
                               img_ncols].reshape((img_nrows, img_ncols))
    target_mask_label = labels[img_nrows *
                               img_ncols:].reshape((img_nrows, img_ncols))

    stack_axis = 0 
    style_mask = np.stack([style_mask_label == r for r in range(nb_labels)],
                          axis=stack_axis)
    target_mask = np.stack([target_mask_label == r for r in range(nb_labels)],
                           axis=stack_axis)

    return (np.expand_dims(style_mask, axis=0),
            np.expand_dims(target_mask, axis=0))


def batch_flatten(x):
    x = T.reshape(x, (x.shape[0], T.prod(x.shape) // x.shape[0]))
    return x

def gram_matrix(x):
    features = batch_flatten(x)
    gram = T.dot(features, T.transpose(features))
    return gram
