from .architecture import *
from ..utils.argchecker import check_type
from ..data.dataset import DataSet

import tensorflow as tf

class Model(Architecture):

    def __init__(self, input_model, layer_models, output_model):
        super().__init__(input_model, layer_models, output_model)
        self.__layers = []

        #self.create_model()

    @property
    def layers(self):
        return self.__layers

    def build(self):

        self.layers.append(self.input(self.layer_models[0]))
        for layer_model in self.layer_models[1:]:
            input = self.layers[-1]
            if isinstance(layer_model, Convolution):
                self.layers.append(self.convolution(input, layer_model))

            elif isinstance(layer_model, MaxPooling):
                self.layers.append(self.maxPooling(input, layer_model))

            elif isinstance(layer_model, Flatten):
                self.layers.append(self.flatten(input, layer_model))

            elif isinstance(layer_model, FullyConnection):
                self.layers.append(self.fully_connection(input, layer_model))

            elif isinstance(layer_model, DropOut):
                self.layers.append(self.dropout(input, layer_model))

            else:
                raise ValueError()

    @property
    def score(self):
        return self.layers[-1]

    def train(self, X, labels, test_X, test_labels, epoch, batch_size):
        dataset = DataSet(X, labels, test_X, test_labels)

        y_true = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

        loss = self.loss_function(dataset.labels)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001)
        train = optimizer.minimize(loss)

        init = tf.compat.v1.global_variables_initializer()

        with tf.Session() as session:
            session.run(init)

            for i in range(epoch):
                print('Epoch: {0}'.format(i))
                for j in range(0, len(dataset.labels), batch_size):

                    x, labels = dataset.batch(batch_size, i)
                    session.run(train, feed_dict={'x': x, 'y_true': labels, 'keep_prob':0.5})

                matches = tf.equal(tf.argmax(self.score, 1), tf.argmax(y_true, 1))
                acc = tf.reduce_mean(tf.cast(matches, tf.float32))

                test_acc = acc.eval(feed_dict={'x': dataset.test_X, 'y_true': dataset.test_labels, 'keep_prob':1.0})
                print('accuracy: {0}'.format(test_acc))

    def loss_function(self, labels):
        # cross entropy
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=self.score))

    """
    below functions may be moved to utils
    """
    def input(self, input):
        check_type(input, 'input', Input, Model, funcnames='input')
        return tf.compat.v1.placeholder(tf.float32, shape=[None, input.height, input.width, 3])

    def convolution(self, input, convolution):
        check_type(convolution, 'convolution', Convolution, Model, funcnames='convolution')
        input_channels = int(input.get_shape()[-1])
        with tf.compat.v1.variable_scope(convolution.name):
            # get variable name's value in scope, if which doesn't exist create it
            weights = tf.compat.v1.get_variable(convolution.name + '_w', shape=[convolution.kernel_height, convolution.kernel_width, input_channels, convolution.kernelnums],
                                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            bias = tf.compat.v1.get_variable(convolution.name + '_b', shape=[convolution.kernelnums],
                                   initializer=tf.constant_initializer(0.0))

            # convolutioln
            wx = tf.nn.conv2d(input, weights, strides=[1, convolution.stride_height, convolution.stride_width, 1], padding=convolution.padding)
            # w*x + b
            output = tf.nn.bias_add(wx, bias)
            return tf.nn.relu(output)

    def maxPooling(self, input, maxpooling):
        check_type(maxpooling, 'maxpooling', MaxPooling, Model, funcnames='maxpooling')
        return tf.nn.max_pool2d(input, ksize=[1, maxpooling.kernel_height, maxpooling.kernel_width, 1],
                              strides=[1, maxpooling.stride_height, maxpooling.stride_width, 1], padding=maxpooling.padding)
    #def batch_normalization(self):

    def flatten(self, input, flatten):
        check_type(flatten, 'flatten', Flatten, Model, funcnames='flatten')
        shape = np.array(input.get_shape().as_list()[1:])
        return tf.reshape(input, [-1, shape.prod()])

    def fully_connection(self, input, fullyconnection):
        check_type(fullyconnection, 'fullyconnection', FullyConnection, Model, funcnames='fullyconnection')

        inputnums = int(input.get_shape()[-1])
        with tf.compat.v1.variable_scope(fullyconnection.name):
            # get variable name's value in scope, if which doesn't exist create it
            weights = tf.compat.v1.get_variable(fullyconnection.name + '_w', shape=[inputnums, fullyconnection.outputnums],
                                      initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))

            bias = tf.compat.v1.get_variable(fullyconnection.name + '_b', shape=[fullyconnection.outputnums], initializer=tf.constant_initializer(0.0))

            wx = tf.matmul(input, weights)
            output = tf.nn.bias_add(wx, bias)

            if fullyconnection.activationfunc is None:
                return output
            elif fullyconnection.activationfunc == 'relu':
                return tf.nn.relu(output)
            else:
                #raise!
                return

    def dropout(self, input, dropout):
        check_type(dropout, 'dropout', DropOut, Model, funcnames='dropout')
        return tf.nn.dropout(input, rate=dropout.rate)