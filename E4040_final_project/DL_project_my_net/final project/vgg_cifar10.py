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
from hw3_utils import load_data
from hw3_utils import load_data_augmentation
from hw3_utils import shared_dataset
from keras.preprocessing.image import load_img, img_to_array

import pickle
import cPickle
import scipy.io 

import matplotlib.pyplot as plt
import random
from keras import backend as K
from keras.applications import vgg16
%matplotlib inline

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

def train_nn(train_model, validate_model, test_model,
                n_train_batches, n_valid_batches, n_test_batches, n_epochs,
                verbose = True):
        """
        Wrapper function for training and test THEANO model

        :type train_model: Theano.function
        :param train_model:

        :type validate_model: Theano.function
        :param validate_model:

        :type test_model: Theano.function
        :param test_model:

        :type n_train_batches: int
        :param n_train_batches: number of training batches

        :type n_valid_batches: int
        :param n_valid_batches: number of validation batches

        :type n_test_batches: int
        :param n_test_batches: number of testing batches

        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer

        :type verbose: boolean
        :param verbose: to print out epoch summary or not to

        """
        
        
        print ('*************** Training ********************')
        # early-stopping parameters
        patience = 100000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.85  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_train_batches, patience // 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.
        start_time = timeit.default_timer()

        epoch = 0
        done_looping = False

        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(n_train_batches):

                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter % 100 == 0) and verbose:
                    print('training @ iter = ', iter)
                cost_ij = train_model(minibatch_index)

                if (iter + 1) % validation_frequency == 0:

                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                         in range(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)

                    if verbose:
                        print('epoch %i, minibatch %i/%i, validation error %f %%' %
                            (epoch,
                             minibatch_index + 1,
                             n_train_batches,
                             this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = [
                            test_model(i)
                            for i in range(n_test_batches)
                        ]
                        test_score = numpy.mean(test_losses)

                        if verbose:
                            print(('epoch %i, minibatch %i/%i, test error of '
                                   'best model %f %%') %
                                  (epoch, minibatch_index + 1,
                                   n_train_batches,
                                   test_score * 100.))

                if patience <= iter:
                    done_looping = True
                    break

        end_time = timeit.default_timer()

        # Retrieve the name of function who invokes train_nn() (caller's name)
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)

        # Print out summary
        print('Optimization complete.')
        print('Best validation error of %f %% obtained at iteration %i, '
              'with test performance %f %%' %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print(('The training process for function ' +
               calframe[1][3] +
               ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).
        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

    
class LeNetConvLayer(object):
        """Pool Layer of a convolutional network """

        def __init__(self, rng, input, filter_shape, image_shape):
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
            self.b = theano.shared(value=b_values, borrow=True)
            
            #self.W = weight
            #self.b = bias
            # convolve input feature maps with filters
            conv_out = conv2d(
                input=input,
                filters=self.W,
                filter_shape=filter_shape,
                input_shape=image_shape
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
            #self.b = theano.shared(value=bias, borrow=True)
            self.b = bias
            # convolve input feature maps with filters
            conv_out = conv2d(
                input=input,
                filters=self.W,
                filter_shape=filter_shape,
                input_shape=image_shape
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
            pooled_out = pool.max_pool_2d_same_size(
                input=input,
                patch_size=poolsize
            )

            self.output = pooled_out
            # store parameters of this layer

            # keep track of model input
            self.input = input



def Adam(cost, params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
    updates = []
    grads = T.grad(cost, params)
    i = theano.shared(numpy.float32(1))
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates   

    
def construct_model_4(dataset='cifar-10-matlab.tar.gz',
                        learning_rate=0.1, 
                        batch_size=200,
                        n_epochs=25):
        
        rng = numpy.random.RandomState(23455)
        print ('loading started')
        ds_rate=None
        

        datasets = load_data(ds_rate=ds_rate,theano_shared=True)
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]
        
        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_train_batches //= batch_size
        n_valid_batches //= batch_size
        n_test_batches //= batch_size
        print ('loading complete')
        
        training_enabled = T.iscalar('training_enabled')
        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch
        
        #theano.config.compute_test_value = 'warn'
        
        # start-snippet-1
        x = T.matrix('x')   # the data is presented as rasterized images
        y = T.ivector('y')  # the labels are presented as 1D vector of
                            # [int] labels

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print('... building the model')

        
        #vgg_rawnet     = scipy.io.loadmat('imagenet-vgg-verydeep-16.mat')
        #vgg_layers = vgg_rawnet['layers'][0]

        layer0_input = x.reshape((batch_size, 3, 32, 32))

        layer1 = LeNetConvLayer(
            rng,
            input=layer0_input,
            image_shape=(batch_size, 3, 
                         32, 
                         32),
            filter_shape=(64, 3, 3, 3)
            #poolsize=pool_size[1]
        )
        
        layer2 = LeNetConvLayer(
            rng,
            input=layer1.output,
            image_shape=(batch_size, 64, 
                         30, 
                         30),
            filter_shape=(64, 64, 3, 3)
            #poolsize=pool_size[1]
        )
        
        
        layer3 = LeNetPoolLayer(
            rng,
            input=layer2.output,
            image_shape=(batch_size, 64, 
                         28, 
                         28),
            #filter_shape=(64, 64, 3, 3)
            poolsize=(2,2)
        )

        
        layer4 = LeNetConvLayer(
            rng,
            input=layer3.output,
            image_shape=(batch_size, 64, 
                         28, 
                         28),
            filter_shape=(128, 64, 3, 3)
            #poolsize=pool_size[1]
        )
        
        layer5 = LeNetConvLayer(
            rng,
            input=layer4.output,
            image_shape=(batch_size, 128, 
                         26, 
                         26),
            filter_shape=(128, 128, 3, 3)
            #poolsize=pool_size[1]
        )
        
        layer6 = LeNetPoolLayer(
            rng,
            input=layer5.output,
            image_shape=(batch_size, 128, 
                         24, 
                         24),
            #filter_shape=(64, 64, 3, 3)
            poolsize=(2,2)
        )
        
        layer7 = LeNetConvLayer(
            rng,
            input=layer6.output,
            image_shape=(batch_size, 128, 
                         24, 
                         24),
            filter_shape=(256, 128, 3, 3)
            #poolsize=pool_size[1]
        )
        
        layer8 = LeNetConvLayer(
            rng,
            input=layer7.output,
            image_shape=(batch_size, 256, 
                         22, 
                         22),
            filter_shape=(256, 256, 3, 3)
            #poolsize=pool_size[1]
        )

	layer9 = LeNetConvLayer(
            rng,
            input=layer8.output,
            image_shape=(batch_size, 256, 
                         20, 
                         20),
            filter_shape=(256, 256, 3, 3)
            #poolsize=pool_size[1]
        )
        
        layer10 = LeNetPoolLayer(
            rng,
            input=layer9.output,
            image_shape=(batch_size, 256, 
                         18, 
                         18),
            #filter_shape=(64, 64, 3, 3)
            poolsize=(2,2)
        )

	layer11 = LeNetConvLayer(
            rng,
            input=layer10.output,
            image_shape=(batch_size, 256, 
                         18, 
                         18),
            filter_shape=(512, 256, 3, 3)
            #poolsize=pool_size[1]
        )
        
        layer12 = LeNetConvLayer(
            rng,
            input=layer11.output,
            image_shape=(batch_size, 512, 
                         16, 
                         16),
            filter_shape=(512, 512, 3, 3)
            #poolsize=pool_size[1]
        )

	layer13 = LeNetConvLayer(
            rng,
            input=layer12.output,
            image_shape=(batch_size, 512, 
                         14, 
                         14),
            filter_shape=(512, 512, 3, 3)
            #poolsize=pool_size[1]
        )
        
        layer14 = LeNetPoolLayer(
            rng,
            input=layer13.output,
            image_shape=(batch_size, 512, 
                         12, 
                         12),
            #filter_shape=(64, 64, 3, 3)
            poolsize=(2,2)
        )
	
	layer15 = LeNetConvLayer(
            rng,
            input=layer14.output,
            image_shape=(batch_size, 512, 
                         12, 
                         12),
            filter_shape=(512, 512, 3, 3)
            #poolsize=pool_size[1]
        )
        
        layer16 = LeNetConvLayer(
            rng,
            input=layer15.output,
            image_shape=(batch_size, 512, 
                         10, 
                         10),
            filter_shape=(512, 512, 3, 3)
            #poolsize=pool_size[1]
        )

	layer17 = LeNetConvLayer(
            rng,
            input=layer16.output,
            image_shape=(batch_size, 512, 
                         8, 
                         8),
            filter_shape=(512, 512, 3, 3)
            #poolsize=pool_size[1]
        )
        
        layer18 = LeNetPoolLayer(
            rng,
            input=layer17.output,
            image_shape=(batch_size, 512, 
                         6, 
                         6),
            #filter_shape=(64, 64, 3, 3)
            poolsize=(2,2)
        )
        
        layerf1_input = layer18.output.flatten(2)
        
        layerf1 = HiddenLayer(
            rng,
            input=layerf1_input,
            n_in=512 * 6 * 6,
            n_out=4096,
            activation=T.nnet.relu
        )
        
        layerf2 = HiddenLayer(
            rng,
            input=layerf1.output,
            n_in=4096,
            n_out=4096,
            activation=T.nnet.relu
        )
        
        # classify the values of the fully-connected sigmoidal layer
        layerf3 = LogisticRegression(input=layerf2.output, n_in=4096, n_out=10)

        # the cost we minimize during training is the NLL of the model
        cost = layerf3.negative_log_likelihood(y)
        

        # create a function to compute the mistakes that are made by the model
        test_model = theano.function(
            [index],
            layerf3.errors(y),
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        validate_model = theano.function(
            [index],
            layerf3.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        # create a list of all model parameters to be fit by gradient descent
        params = layer1.params + layer2.params + layer4.params + layer5.params + layer7.params + \
                 layer8.params + layer9.params + layer11.params + layer12.params + layer13.params + \
		 layer15.params + layer16.params + layer17.params + layerf1.params + layerf2.params + layerf3.params
        #layer10.params + layer20.params + layer00.params 

        # create a list of gradients for all model parameters
        #grads = T.grad(cost, params)

        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i], grads[i]) pairs.
        '''updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(params, grads)
        ]  '''
        
        
        updates = Adam(cost, params)

        train_model = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )
        
        
        '''nimb = MSE(layer9.output)
        cost = nimb.mse(x.reshape((batch_size, 3, 32, 32)))
        

        # create a function to compute the mistakes that are made by the model
        test_model = theano.function(
            [index],
            cost,
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size]
            }
        )

        validate_model = theano.function(
            [index],
            cost,
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size]
            }
        )

        # create a list of all model parameters to be fit by gradient descent
        params = layer1.params +  layer2.params + layer4.params + layer5.params + layer7.params + layer8.params
        #layer10.params + layer20.params + layer00.params 

        # create a list of gradients for all model parameters
        #grads = T.grad(cost, params)

        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i], grads[i]) pairs.
        ''updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(params, grads)
        ] '' 
        
        
        updates = Adam(cost, params)

        train_model = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size]
            }
        ) '''
        
        print ('model built')
        train_nn(train_model, validate_model, test_model,
            n_train_batches, n_valid_batches, n_test_batches, n_epochs,
            verbose = True)
        print ("Complete .... saving")
        best_model = 1
        if (best_model == 1):
            print('Saving Model')
            save_file = open('weights10', 'wb')
            cPickle.dump(layer1.params[0].get_value(borrow=True), save_file, -1)
            save_file.close() 
            save_file = open('weights11', 'wb')
            cPickle.dump(layer1.params[1].get_value(borrow=True), save_file, -1)
            save_file.close() 
            save_file = open('weights20', 'wb')
            cPickle.dump(layer2.params[0].get_value(borrow=True), save_file, -1)
            save_file.close() 
            save_file = open('weights21', 'wb')
            cPickle.dump(layer2.params[1].get_value(borrow=True), save_file, -1)
            save_file.close() 
            save_file = open('weights40', 'wb')
            cPickle.dump(layer4.params[0].get_value(borrow=True), save_file, -1)
            save_file.close() 
            save_file = open('weights41', 'wb')
            cPickle.dump(layer4.params[1].get_value(borrow=True), save_file, -1)
            save_file.close() 
            save_file = open('weights50', 'wb')
            cPickle.dump(layer5.params[0].get_value(borrow=True), save_file, -1)
            save_file.close() 
            save_file = open('weights51', 'wb')
            cPickle.dump(layer5.params[1].get_value(borrow=True), save_file, -1)
            save_file.close() 
            save_file = open('weights70', 'wb')
            cPickle.dump(layer7.params[0].get_value(borrow=True), save_file, -1)
            save_file.close() 
            save_file = open('weights71', 'wb')
            cPickle.dump(layer7.params[1].get_value(borrow=True), save_file, -1)
            save_file.close()
            save_file = open('weights80', 'wb')
            cPickle.dump(layer8.params[0].get_value(borrow=True), save_file, -1)
            save_file.close() 
            save_file = open('weights81', 'wb')
            cPickle.dump(layer8.params[1].get_value(borrow=True), save_file, -1)
            save_file.close() 
	    save_file = open('weights90', 'wb')
            cPickle.dump(layer9.params[0].get_value(borrow=True), save_file, -1)
            save_file.close() 
            save_file = open('weights91', 'wb')
            cPickle.dump(layer9.params[1].get_value(borrow=True), save_file, -1)
            save_file.close() 
	    save_file = open('weights110', 'wb')
            cPickle.dump(layer11.params[0].get_value(borrow=True), save_file, -1)
            save_file.close() 
            save_file = open('weights111', 'wb')
            cPickle.dump(layer11.params[1].get_value(borrow=True), save_file, -1)
            save_file.close() 
	    save_file = open('weights120', 'wb')
            cPickle.dump(layer12.params[0].get_value(borrow=True), save_file, -1)
            save_file.close() 
            save_file = open('weights121', 'wb')
            cPickle.dump(layer12.params[1].get_value(borrow=True), save_file, -1)
            save_file.close() 
	    save_file = open('weights130', 'wb')
            cPickle.dump(layer13.params[0].get_value(borrow=True), save_file, -1)
            save_file.close() 
            save_file = open('weights131', 'wb')
            cPickle.dump(layer13.params[1].get_value(borrow=True), save_file, -1)
            save_file.close() 
	    save_file = open('weights150', 'wb')
            cPickle.dump(layer15.params[0].get_value(borrow=True), save_file, -1)
            save_file.close() 
            save_file = open('weights151', 'wb')
            cPickle.dump(layer15.params[1].get_value(borrow=True), save_file, -1)
            save_file.close() 
	    save_file = open('weights160', 'wb')
            cPickle.dump(layer16.params[0].get_value(borrow=True), save_file, -1)
            save_file.close() 
            save_file = open('weights161', 'wb')
            cPickle.dump(layer16.params[1].get_value(borrow=True), save_file, -1)
            save_file.close() 
	    save_file = open('weights170', 'wb')
            cPickle.dump(layer17.params[0].get_value(borrow=True), save_file, -1)
            save_file.close() 
            save_file = open('weights171', 'wb')
            cPickle.dump(layer17.params[1].get_value(borrow=True), save_file, -1)
            save_file.close() 

def MY_CNN():
	construct_model_4()

MY_CNN()
