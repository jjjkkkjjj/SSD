from ..data.dataset import DataSet
from ssd.params.loss_function import *
from ssd.params.training import TrainingParams
from ..utils.error.argchecker import *

import tensorflow as tf

class OptimezerMixin:
    score: tf.Tensor
    params: TrainingParams

    def __init__(self):
        self.params = None


    def train(self, X, labels, test_X, test_labels, params):
        dataset = DataSet(X, labels, test_X, test_labels)

        self.params = check_type(params, 'params', TrainingParams, OptimezerMixin, funcnames='train')

        iter_params = self.params.iteration
        loss_params = self.params.loss
        opt_params = self.params.optimization

        y_true = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

        loss = multinominal_logistic_reggression(dataset.labels, self.score)

        """
        loss = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=out_layer, labels=tf_train_labels)) +
        0.01*tf.nn.l2_loss(hidden_weights) +
        0.01*tf.nn.l2_loss(out_weights) +
        """

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001)
        train = optimizer.minimize(loss)

        init = tf.compat.v1.global_variables_initializer()

        with tf.compat.v1.Session() as session:
            session.run(init)

            for i in range(iter_params.epoch):
                print('Epoch: {0}'.format(i))
                for j in range(0, len(dataset.labels), iter_params.batch_size):
                    x, labels = dataset.batch(iter_params.batch_size, i)
                    session.run(train, feed_dict={'x': x, 'y_true': labels, 'keep_prob': 0.5})

                matches = tf.equal(tf.argmax(self.score, 1), tf.argmax(y_true, 1))
                acc = tf.reduce_mean(tf.cast(matches, tf.float32))

                test_acc = acc.eval(feed_dict={'x': dataset.test_X, 'y_true': dataset.test_labels, 'keep_prob': 1.0})
                print('accuracy: {0}'.format(test_acc))

