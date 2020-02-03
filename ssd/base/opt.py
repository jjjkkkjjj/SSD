from ..data.dataset import DataSet
from ..params.loss_function import get_loss_function
from ..params.regularization import get_loss_added_regularization
from ..params.training import TrainingParams
from ..data.iterator import *
from ..utils.error.argchecker import *

import tensorflow as tf
import logging

class OptimezerMixin:
    input_layer: tf.compat.v1.placeholder
    score: tf.Tensor
    weights: list
    params: TrainingParams

    def train(self, dataset, params):
        dataset = check_type(dataset, 'dataset', DataSet, OptimezerMixin, funcnames='train')
        dataset: DatasetClassification
        # get params for training
        self.params = check_type(params, 'params', TrainingParams, OptimezerMixin, funcnames='train')

        iter_params = self.params.iter_params
        loss_params = self.params.lossfunc_params
        opt_params = self.params.opt_params

        # define variable without value in train
        input = self.input_layer
        # must be variablized
        y_true = tf.compat.v1.placeholder(tf.float32, shape=[None, dataset.class_num])


        """
        loss = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=out_layer, labels=tf_train_labels)) +
        0.01*tf.nn.l2_loss(hidden_weights) +
        0.01*tf.nn.l2_loss(out_weights) +
        """

        # set objective function
        loss = get_loss_function(y_true, self.score, loss_params, OptimezerMixin)
        loss = get_loss_added_regularization(self.weights, loss, loss_params, OptimezerMixin)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=opt_params.learning_rate)
        objective_function = optimizer.minimize(loss)



        init = tf.compat.v1.global_variables_initializer()

        # training
        with tf.compat.v1.Session() as session:
            with tf.name_scope('summary'):
                tf.compat.v1.summary.scalar('loss', loss)
                merged = tf.compat.v1.summary.merge_all()
                writer = tf.compat.v1.summary.FileWriter('./logs', session.graph)

            session.run(init)

            for epoch in dataset.epoch_iterator(iter_params, random_by_epoch=True):
                epoch: EpochIteratorClassification
                logging.info('\nEpoch: {0}\n'.format(epoch.epoch_now))

                for batch in epoch.batch_iterator():
                    batch: BatchIteratorClassification
                    logging.info('batch: {0}/{1}'.format(batch.iteration_now, batch.iteration))
                    session.run(objective_function, feed_dict={input: batch.X, y_true: batch.one_hotted_labels})# 'keep_prob': 1.0 see https://github.com/Natsu6767/VGG16-Tensorflow/blob/master/vgg16.py

                matches = tf.equal(tf.argmax(self.score, 1), tf.argmax(y_true, 1))
                acc = tf.reduce_mean(tf.cast(matches, tf.float32))

                test_acc = acc.eval(feed_dict={'x': dataset.test_X, 'y_true': dataset.test_one_hotted_labels}) # 'keep_prob': 1.0 see https://github.com/Natsu6767/VGG16-Tensorflow/blob/master/vgg16.py
                logging.info('accuracy: {0}'.format(test_acc))


            """
            for i in range(iter_params.epoch):
                logging.info('\nEpoch: {0}\n'.format(i))

                for j in range(0, len(dataset.labels), iter_params.batch_size):
                    x, labels = dataset.batch(iter_params.batch_size, i)
                    session.run(train, feed_dict={'x': x, 'y_true': labels, 'keep_prob': 0.5})

                matches = tf.equal(tf.argmax(self.score, 1), tf.argmax(y_true, 1))
                acc = tf.reduce_mean(tf.cast(matches, tf.float32))

                test_acc = acc.eval(feed_dict={'x': dataset.test_X, 'y_true': dataset.test_labels, 'keep_prob': 1.0})
                logging.info('accuracy: {0}'.format(test_acc))
            """

