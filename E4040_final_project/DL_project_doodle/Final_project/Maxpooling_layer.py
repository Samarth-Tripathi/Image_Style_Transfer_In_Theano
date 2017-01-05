import os
import sys
import numpy
import scipy.io
import tarfile
import theano
import theano.tensor as T
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


class LeNetPoolLayer(object):
        """Pool Layer of a convolutional network """

        def __init__(self, rng, input, image_shape, name, poolsize=(2, 2)):
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
            pooled_out = pool.max_pool_2d_same_size(
                input=input,
                patch_size=poolsize
            )

            self.output = pooled_out
            # store parameters of this layer
            
            self.name = name

            # keep track of model input
            self.input = input

class LeNetPoolLayer2(object):
        """Pool Layer of a convolutional network """

        def __init__(self, rng, input, image_shape, name, poolsize=(2, 2)):
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
                #mode = 'avg_pad_incl'
                ignore_border=True
            )
            
            self.name = name
            self.output = pooled_out
            # store parameters of this layer

            # keep track of model input
            self.input = input

def pool_model(input):

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print('... building the model')
        
        rng = numpy.random.RandomState(1234)
        batch_size = 2
        layer0_input = input.reshape((batch_size, 4, 400, 400))
        pool_size = [(2,2),(3,3)]

        layer1 = LeNetPoolLayer(
            rng,
            input=layer0_input,
            image_shape=(batch_size, 4, 
                         400, 
                         400),
            name = 'mask_1',
            poolsize=pool_size[1]
        )
        
        layer2 = LeNetPoolLayer(
            rng,
            input=layer1.output,
            image_shape=(batch_size, 4, 
                         400, 
                         400),
            name = 'mask_2',
            poolsize=pool_size[1]
        )
        
        
        layer3 = LeNetPoolLayer2(
            rng,
            input=layer2.output,
            image_shape=(batch_size, 4, 
                         400, 
                         400),
            name = 'mask_3',
            poolsize=pool_size[0]
        )

        
        layer4 = LeNetPoolLayer(
            rng,
            input=layer3.output,
            image_shape=(batch_size, 4, 
                         200, 
                         200),
            name = 'mask_4',
            poolsize=pool_size[1]
        )
        
        layer5 = LeNetPoolLayer(
            rng,
            input=layer4.output,
            image_shape=(batch_size, 4, 
                         200, 
                         200),
            name = 'mask_5',
            poolsize=pool_size[1]
        )
        
        
        layer6 = LeNetPoolLayer2(
            rng,
            input=layer5.output,
            image_shape=(batch_size, 4, 
                         200, 
                         200),
            name = 'mask_6',
            poolsize=pool_size[0]
        )
        
        layer7 = LeNetPoolLayer(
            rng,
            input=layer6.output,
            image_shape=(batch_size, 4, 
                         100, 
                         100),
            name = 'mask_7',
            poolsize=pool_size[1]
        )
    
        layer8 = LeNetPoolLayer(
            rng,
            input=layer7.output,
            image_shape=(batch_size, 4, 
                         100, 
                         100),
            name = 'mask_8',
            poolsize=pool_size[1]
        )
        
        layer9 = LeNetPoolLayer(
            rng,
            input=layer8.output,
            image_shape=(batch_size, 4, 
                         100, 
                         100),
            name = 'mask_9',
            poolsize=pool_size[1]
        )
        
        
        layer10 = LeNetPoolLayer2(
            rng,
            input=layer9.output,
            image_shape=(batch_size, 4, 
                         100, 
                         100),
            name = 'mask_10',
            poolsize=pool_size[0]
        )
    
        layer11 = LeNetPoolLayer(
            rng,
            input=layer10.output,
            image_shape=(batch_size, 4, 
                         50, 
                         50),
            name = 'mask_11',
            poolsize=pool_size[1]
        )
    
        layer12 = LeNetPoolLayer(
            rng,
            input=layer11.output,
            image_shape=(batch_size, 4, 
                         50, 
                         50),
            name = 'mask_12',
            poolsize=pool_size[1]
        )
        
        layer13 = LeNetPoolLayer(
            rng,
            input=layer12.output,
            image_shape=(batch_size, 4, 
                         50, 
                         50),
            name = 'mask_13',
            poolsize=pool_size[1]
        )
        
        
        layer14 = LeNetPoolLayer2(
            rng,
            input=layer13.output,
            image_shape=(batch_size, 4, 
                         50, 
                         50),
            name = 'mask_14',
            poolsize=pool_size[0]
        )
    
        layer15 = LeNetPoolLayer(
            rng,
            input=layer14.output,
            image_shape=(batch_size, 4, 
                         25, 
                         25),
            name = 'mask_15',
            poolsize=pool_size[1]
        )
    
        layer16 = LeNetPoolLayer(
            rng,
            input=layer15.output,
            image_shape=(batch_size, 4, 
                         25, 
                         25),
            name = 'mask_16',
            poolsize=pool_size[1]
        )
        
        layer17 = LeNetPoolLayer(
            rng,
            input=layer16.output,
            image_shape=(batch_size, 4, 
                         25, 
                         25),
            name = 'mask_17',
            poolsize=pool_size[1]
        )
        
        
        layer18 = LeNetPoolLayer2(
            rng,
            input=layer17.output,
            image_shape=(batch_size, 4, 
                         25, 
                         25),
            name = 'mask_18',
            poolsize=pool_size[0]
        )

        x=T.iscalar('x')
        execute_model = theano.function(
            [x],
     [layer1.output,layer2.output,layer3.output,layer4.output,layer5.output,layer6.output,layer7.output,layer8.output,layer9.output,layer10.output,layer11.output,layer12.output,layer13.output,layer14.output,layer15.output,layer16.output,layer17.output,layer18.output],
            on_unused_input='warn'
        )
        
        return execute_model(1)