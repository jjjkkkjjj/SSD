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
        weights = tf.compat.v1.get_variable(convolution.name + '_w',
                                            shape=[convolution.kernel_height, convolution.kernel_width, input_channels,
                                                   convolution.kernelnums],
                                            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        bias = tf.compat.v1.get_variable(convolution.name + '_b', shape=[convolution.kernelnums],
                                         initializer=tf.constant_initializer(0.0))

        # convolutioln
        wx = tf.nn.conv2d(input, weights, strides=[1, convolution.stride_height, convolution.stride_width, 1],
                          padding=convolution.padding)
        # w*x + b
        output = tf.nn.bias_add(wx, bias)
        return tf.nn.relu(output)


def maxPooling(input, maxpooling):
    from ..base.model import Model

    check_type(maxpooling, 'maxpooling', MaxPooling, Model, funcnames='maxpooling')
    return tf.nn.max_pool2d(input, ksize=[1, maxpooling.kernel_height, maxpooling.kernel_width, 1],
                            strides=[1, maxpooling.stride_height, maxpooling.stride_width, 1],
                            padding=maxpooling.padding)


# def batch_normalization(self):

def flatten(input, flatten):
    from ..base.model import Model

    check_type(flatten, 'flatten', Flatten, Model, funcnames='flatten')
    shape = np.array(input.get_shape().as_list()[1:])
    return tf.reshape(input, [-1, shape.prod()])


def fully_connection(input, fullyconnection):
    from ..base.model import Model

    check_type(fullyconnection, 'fullyconnection', FullyConnection, Model, funcnames='fullyconnection')

    inputnums = int(input.get_shape()[-1])
    with tf.compat.v1.variable_scope(fullyconnection.name):
        # get variable name's value in scope, if which doesn't exist create it
        weights = tf.compat.v1.get_variable(fullyconnection.name + '_w', shape=[inputnums, fullyconnection.outputnums],
                                            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))

        bias = tf.compat.v1.get_variable(fullyconnection.name + '_b', shape=[fullyconnection.outputnums],
                                         initializer=tf.constant_initializer(0.0))

        wx = tf.matmul(input, weights)
        output = tf.nn.bias_add(wx, bias)

        if fullyconnection.activationfunc is None:
            return output
        elif fullyconnection.activationfunc == 'relu':
            return tf.nn.relu(output)
        else:
            # raise!
            return


def dropout(input, dropout):
    from ..base.model import Model

    check_type(dropout, 'dropout', DropOut, Model, funcnames='dropout')
    return tf.nn.dropout(input, rate=dropout.rate)