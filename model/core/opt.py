from ..train.loss_function import get_loss_function
from ..train.regularization import get_loss_added_regularization
from ..train.params import TrainingParams
from ..dataset.iterator import *
from ..common.utils.argchecker import *

import tensorflow as tf
import logging

class OptimezerMixin:
    input_layer: tf.compat.v1.placeholder
    score: tf.Tensor
    weights: list
    params: TrainingParams

    def train(self, dataset, params):
        dataset = check_type(dataset, 'dataset', DataSet, self, funcnames='train')

        # type must be checked
        dataset: DatasetClassification
        # get params for training
        self.params = check_type(params, 'params', TrainingParams, self, funcnames='train')

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
        loss = get_loss_function(y_true, self.score, loss_params, self)
        loss = get_loss_added_regularization(self.weights, loss, loss_params, self)
        optimizer = opt_params.optimizer
        objective_function = optimizer.minimize(loss)

        # set accuracy
        matches = tf.equal(tf.argmax(self.score, 1), tf.argmax(y_true, 1))
        accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))

        init = tf.compat.v1.global_variables_initializer()

        # training
        with tf.compat.v1.Session() as session:
            with tf.name_scope('summary'):
                tf.compat.v1.summary.scalar('loss', loss)
                merged = tf.compat.v1.summary.merge_all()
                writer = tf.compat.v1.summary.FileWriter('./logs', session.graph)

            session.run(init)

            for epoch in dataset.epoch_iterator(opt_params):
                epoch: EpochIteratorClassification
                logging.info('\nEpoch: {0}\n'.format(epoch.epoch_now))

                for batch in epoch.batch_iterator():
                    batch: BatchIteratorClassification
                    # optimize objective function
                    session.run(objective_function, feed_dict={input: batch.X, y_true: batch.one_hotted_labels})# 'keep_prob': 1.0 see https://github.com/Natsu6767/VGG16-Tensorflow/blob/master/vgg16.py
                    # get loss value and train accuracy
                    loss_val, train_acc = session.run([loss, accuracy], feed_dict={input: batch.X, y_true: batch.one_hotted_labels})

                    score, true, m = session.run([self.score, y_true, matches],
                                                      feed_dict={input: batch.X, y_true: batch.one_hotted_labels})
                    #print(score, true, m)
                    logging.info('batch: {0}/{1}, loss: {2:.2f}, train accuracy: {3:.2f}'.format(batch.iteration_now, batch.iteration, loss_val, train_acc))


                loss_val, test_acc = session.run([loss, accuracy], feed_dict={input: epoch.test_X, y_true: epoch.test_one_hotted_labels})
                #test_acc = acc.eval(feed_dict={input: dataset.test_X, y_true: dataset.test_one_hotted_labels}) # 'keep_prob': 1.0 see https://github.com/Natsu6767/VGG16-Tensorflow/blob/master/vgg16.py
                logging.info('\nepoch: {0}/{1}, loss: {2:2f}, test accuracy: {3:.2f}\n'.format(epoch.epoch_now, epoch.epoch, loss_val, test_acc))

