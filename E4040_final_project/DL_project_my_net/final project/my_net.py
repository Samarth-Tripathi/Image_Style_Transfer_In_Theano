"""
Source Code for Homework 3 of ECBM E4040, Fall 2016, Columbia University

Instructor: Prof. Zoran Kostic

This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
"""
from __future__ import print_function
import timeit
import inspect
import sys
import scipy 
import numpy

import theano
import theano.tensor as T
from theano import pp

from theano.tensor.nnet import conv
from theano.tensor.nnet import conv2d
#from theano.tensor.nnet import relu
from theano.tensor.signal import pool
from theano.tensor.signal import downsample


import pickle
import cPickle
import scipy.io 

import matplotlib.pyplot as plt
import random
#from keras import backend as K
#from keras.applications import vgg16



            
class LeNetConvLayer2(object):
        """Pool Layer of a convolutional network """

        def __init__(self, rng, input, filter_shape, image_shape, weight, bias):
            """
            Allocate a LeNetConvPoolLayer with shared variable internal parameters.

            :type rng: numpy.random.RandomState
            :param rng: a random number generator used to initialize weights

            :type input: theano.tensor.dtensor4
            :param input: symbolic image tensor, of shape image_shape

            :type filter_shape: tuple or list of length 4
            :param filter_shape: (number of filters, num input feature maps,
                                  filter height, filter width)

            :type image_shape: tuple or list of length 4
            :param image_shape: (batch size, num input feature maps,
                                 image height, image width)

            :type poolsize: tuple or list of length 2
            :param poolsize: the downsampling (pooling) factor (#rows, #cols)
            """

            assert image_shape[1] == filter_shape[1]
            self.input = input

            # there are "num input feature maps * filter height * filter width"
            # inputs to each hidden unit
            fan_in = numpy.prod(filter_shape[1:])
            # each unit in the lower layer receives a gradient from:
            # "num output feature maps * filter height * filter width" /
            #   pooling size
            fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) )
            # initialize weights with random weights
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

            # the bias is a 1D tensor -- one bias per output feature map
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            #self.b = theano.shared(value=b_values, borrow=True)
            
            self.W = weight
            self.b = theano.shared(value=bias, borrow=True)
            # convolve input feature maps with filters
            conv_out = conv2d(
                input=input,
                filters=self.W,
                filter_shape=filter_shape,
                input_shape=image_shape,
                border_mode = 'half'
            )

            # add the bias term. Since the bias is a vector (1D array), we first
            # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
            # thus be broadcasted across mini-batches and feature map
            # width & height
            self.output = T.nnet.relu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        
            # store parameters of this layer
            self.params = [self.W, self.b]

            # keep track of model input
            self.input = input

class LeNetPoolLayer(object):
        """Pool Layer of a convolutional network """

        def __init__(self, rng, input, image_shape, poolsize=(2, 2)):
            """
            Allocate a LeNetConvPoolLayer with shared variable internal parameters.

            :type rng: numpy.random.RandomState
            :param rng: a random number generator used to initialize weights

            :type input: theano.tensor.dtensor4
            :param input: symbolic image tensor, of shape image_shape

            :type filter_shape: tuple or list of length 4
            :param filter_shape: (number of filters, num input feature maps,
                                  filter height, filter width)

            :type image_shape: tuple or list of length 4
            :param image_shape: (batch size, num input feature maps,
                                 image height, image width)

            :type poolsize: tuple or list of length 2
            :param poolsize: the downsampling (pooling) factor (#rows, #cols)
            """

            self.input = input
            
            # pool each feature map individually, using maxpooling
            pooled_out = pool.pool_2d(
                input=input,
                ds=poolsize,
                ignore_border=True
            )

            self.output = pooled_out
            # store parameters of this layer

            # keep track of model input
            self.input = input


            
def load_saved_model(input, learning_rate=0.1, 
                        batch_size=200):
        
        rng = numpy.random.RandomState(23455)
        print ('loading started')

        #vgg_rawnet     = scipy.io.loadmat('imagenet-vgg-verydeep-16.mat')
        #vgg_layers = vgg_rawnet['layers'][0]
        
        save_file = open('weights10')
        p10 = cPickle.load(save_file)
        save_file = open('weights11')
        p11 = cPickle.load(save_file)
        save_file = open('weights20')
        p20 = cPickle.load(save_file)
        save_file = open('weights21')
        p21 = cPickle.load(save_file)
        save_file = open('weights40')
        p40 = cPickle.load(save_file)
        save_file = open('weights41')
        p41 = cPickle.load(save_file)
        save_file = open('weights50')
        p50 = cPickle.load(save_file)
        save_file = open('weights51')
        p51 = cPickle.load(save_file)
        save_file = open('weights70')
        p70 = cPickle.load(save_file)
        save_file = open('weights71')
        p71 = cPickle.load(save_file)
        save_file = open('weights80')
        p80 = cPickle.load(save_file)
        save_file = open('weights81')
        p81 = cPickle.load(save_file)
        
        print ('loading complete')

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print('... building the model')
        
        #vgg_rawnet     = scipy.io.loadmat('imagenet-vgg-verydeep-16.mat')
        #vgg_layers = vgg_rawnet['layers'][0]
        batch_size = 1
        layer0_input = input   #.reshape((batch_size, 3, 32, 32))

        layer1 = LeNetConvLayer2(
            rng,
            input=layer0_input,
            image_shape=(batch_size, 3, 
                         32, 
                         32),
            filter_shape=(64, 3, 3, 3),
            weight=p10,
            bias=p11
            #poolsize=pool_size[1]
        )
        
        layer2 = LeNetConvLayer2(
            rng,
            input=layer1.output,
            image_shape=(batch_size, 64, 
                         32, 
                         32),
            filter_shape=(64, 64, 3, 3),
            weight=p20,
            bias=p21
            #poolsize=pool_size[1]
        )
        
        
        layer3 = LeNetPoolLayer(
            rng,
            input=layer2.output,
            image_shape=(batch_size, 64, 
                         32, 
                         32),
            #filter_shape=(64, 64, 3, 3)
            poolsize=(2,2)
        )
        
        #l3_out = layer3.output.reshape((64,16,16))
        l3_out = layer3.output
        layer4 = LeNetConvLayer2(
            rng,
            input=layer3.output,
            image_shape=(batch_size, 64, 
                         16, 
                         16),
            filter_shape=(128, 64, 3, 3),
            weight=p40,
            bias=p41
            #poolsize=pool_size[1]
        )
        
        layer5 = LeNetConvLayer2(
            rng,
            input=layer4.output,
            image_shape=(batch_size, 128, 
                         16, 
                         16),
            filter_shape=(128, 128, 3, 3),
            weight=p50,
            bias=p51
            #poolsize=pool_size[1]
        )
        
        layer6 = LeNetPoolLayer(
            rng,
            input=layer5.output,
            image_shape=(batch_size, 128, 
                         16, 
                         16),
            #filter_shape=(64, 64, 3, 3)
            poolsize=(2,2)
        )
        
        #l6_out = layer6.output.reshape((128,8,8))
        l6_out = layer6.output
        layer7 = LeNetConvLayer2(
            rng,
            input=layer6.output,
            image_shape=(batch_size, 128, 
                         8, 
                         8),
            filter_shape=(256, 128, 3, 3),
            weight=p70,
            bias=p71
            #poolsize=pool_size[1]
        )
        
        layer8 = LeNetConvLayer2(
            rng,
            input=layer7.output,
            image_shape=(batch_size, 256, 
                         8, 
                         8),
            filter_shape=(256, 256, 3, 3),
            weight=p80,
            bias=p81
            #poolsize=pool_size[1]
        )
        
        layer9 = LeNetPoolLayer(
            rng,
            input=layer8.output,
            image_shape=(batch_size, 256, 
                         8, 
                         8),
            #filter_shape=(64, 64, 3, 3)
            poolsize=(2,2)
        )
        
        #l9_out = layer9.output.reshape((256,4,4))
        l9_out = layer9.output
        '''x=T.iscalar('x')
        execute_model = theano.function(
            [x],
            [l3_out,l6_out,l9_out],
            on_unused_input='warn'
        )
        
        return execute_model(1)'''
        return [l3_out,l6_out,l9_out]
        
        #return [[l3_out[0],l6_out[0],l9_out[0]],[l3_out[0],l6_out[0],l9_out[0]],[l3_out[0],l6_out[0],l9_out[0]]]
        

#Problem4
#Implement the convolutional neural network depicted in problem4 
'''def MY_CNN():
    #construct_model_4()
    dataset='cifar-10-matlab.tar.gz'
    datasets = load_data(ds_rate=None,theano_shared=False)
    train_set_x = datasets[0][0]
    pic = train_set_x[15]
    #plt.imshow(numpy.reshape(pic,(3,32,32)).transpose(1,2,0))
    p=load_saved_model(input = pic)
    p = p[2].reshape(256,4,4)
    
    f, axarr = plt.subplots(2,2)
    for i in range(2):
        for j in range(2):
            plt.axes(axarr[i,j])
            plt.imshow(p[i*0+j])'''

