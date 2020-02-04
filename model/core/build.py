from .architecture import *
from ..common.utils.typename import _get_typename

import tensorflow as tf

def input(input):
    #from ..core.model import Model

    #check_type(input, 'input', Input, Model, funcnames='input')
    assert isinstance(input, Input), 'got {0}'.format(_get_typename(input))
    with tf.compat.v1.variable_scope(input.name):
        shape = [None]
        shape.extend(input.shape)
        return tf.compat.v1.placeholder(tf.float32, shape=shape)


def convolution(input, convolution):
    #from ..core.model import Model

    #check_type(convolution, 'convolution', Convolution, Model, funcnames='convolution')
    assert isinstance(convolution, Convolution), 'got {0}'.format(_get_typename(convolution))
    input_channels = int(input.get_shape()[-1])
    with tf.compat.v1.variable_scope(convolution.name):
        # get variable name's value in scope, if which doesn't exist create it
        weights = tf.compat.v1.get_variable('Weights',
                                            shape=[convolution.kernel_height, convolution.kernel_width, input_channels,
                                                   convolution.kernelnums],
                                            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        bias = tf.compat.v1.get_variable('Bias', shape=[convolution.kernelnums],
                                         initializer=tf.constant_initializer(0.0))

        # convolutioln
        wx = tf.nn.conv2d(input, weights, strides=[1, convolution.stride_height, convolution.stride_width, 1],
                          padding=convolution.padding)
        # w*x + b
        output = tf.nn.bias_add(wx, bias)
        return tf.nn.relu(output), weights, bias


def maxPooling(input, maxpooling):
    #from ..core.model import Model

    #check_type(maxpooling, 'maxpooling', MaxPooling, Model, funcnames='maxpooling')
    assert isinstance(maxpooling, MaxPooling), 'got {0}'.format(_get_typename(maxpooling))
    with tf.compat.v1.variable_scope(maxpooling.name):
        return tf.nn.max_pool2d(input, ksize=[1, maxpooling.kernel_height, maxpooling.kernel_width, 1],
                            strides=[1, maxpooling.stride_height, maxpooling.stride_width, 1],
                            padding=maxpooling.padding)


# def batch_normalization(self):

def flatten(input, flatten):
    #from ..core.model import Model

    #check_type(flatten, 'flatten', Flatten, Model, funcnames='flatten')
    assert isinstance(flatten, Flatten), 'got {0}'.format(_get_typename(flatten))
    with tf.compat.v1.variable_scope(flatten.name):
        shape = np.array(input.get_shape().as_list()[1:])
        return tf.reshape(input, [-1, shape.prod()], name='Flatten')


def fully_connection(input, fullyconnection):
    #from ..core.model import Model

    #check_type(fullyconnection, 'fullyconnection', FullyConnection, Model, funcnames='fullyconnection')
    assert isinstance(fullyconnection, FullyConnection), 'got {0}'.format(_get_typename(fullyconnection))

    inputnums = int(input.get_shape()[-1])
    with tf.compat.v1.variable_scope(fullyconnection.name):
        # get variable name's value in scope, if which doesn't exist create it
        weights = tf.compat.v1.get_variable('Weights', shape=[inputnums, fullyconnection.outputnums],
                                            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))

        bias = tf.compat.v1.get_variable('Bias', shape=[fullyconnection.outputnums],
                                         initializer=tf.constant_initializer(0.0))

        wx = tf.matmul(input, weights)
        output = tf.nn.bias_add(wx, bias)

        """
        below must be moved to argcheck
        """
        if fullyconnection.activationfunc is None:
            return output, weights, bias
        elif fullyconnection.activationfunc == 'relu':
            return tf.nn.relu(output), weights, bias
        elif fullyconnection.activationfunc == 'softmax':
            return tf.nn.softmax(output), weights, bias
        else:
            # raise!
            return


def dropout(input, dropout):
    #from ..core.model import Model

    #check_type(dropout, 'dropout', DropOut, Model, funcnames='dropout')
    assert isinstance(dropout, DropOut), 'got {0}'.format(_get_typename(dropout))

    with tf.compat.v1.variable_scope(dropout.name):
        return tf.nn.dropout(input, rate=dropout.rate, name='Dropout')