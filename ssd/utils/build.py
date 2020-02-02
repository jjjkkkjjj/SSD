from ..base.architecture import *
from ..utils.error.argchecker import check_type

import tensorflow as tf

def input(input):
    from ..base.model import Model

    check_type(input, 'input', Input, Model, funcnames='input')
    with tf.compat.v1.variable_scope(input.name):
        return tf.compat.v1.placeholder(tf.float32, shape=[None, input.height, input.width, 3])


def convolution(input, convolution):
    from ..base.model import Model

    check_type(convolution, 'convolution', Convolution, Model, funcnames='convolution')
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
    from ..base.model import Model

    check_type(maxpooling, 'maxpooling', MaxPooling, Model, funcnames='maxpooling')
    with tf.compat.v1.variable_scope(maxpooling.name):
        return tf.nn.max_pool2d(input, ksize=[1, maxpooling.kernel_height, maxpooling.kernel_width, 1],
                            strides=[1, maxpooling.stride_height, maxpooling.stride_width, 1],
                            padding=maxpooling.padding)


# def batch_normalization(self):

def flatten(input, flatten):
    from ..base.model import Model

    check_type(flatten, 'flatten', Flatten, Model, funcnames='flatten')
    with tf.compat.v1.variable_scope(flatten.name):
        shape = np.array(input.get_shape().as_list()[1:])
        return tf.reshape(input, [-1, shape.prod()], name='Flatten')


def fully_connection(input, fullyconnection):
    from ..base.model import Model

    check_type(fullyconnection, 'fullyconnection', FullyConnection, Model, funcnames='fullyconnection')

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
        else:
            # raise!
            return


def dropout(input, dropout):
    from ..base.model import Model

    check_type(dropout, 'dropout', DropOut, Model, funcnames='dropout')
    with tf.compat.v1.variable_scope(dropout.name):
        return tf.nn.dropout(input, rate=dropout.rate, name='Dropout')