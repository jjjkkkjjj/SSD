import tensorflow as tf
from tensorflow.keras import metrics
import time

"""
    ref: https://qiita.com/imsk/items/d14be2a7fcca8a080a8d
"""
class Trainer(object):
    def __init__(self, model, loss_func, optimizer):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer

        self.train_loss = metrics.Mean(name='train_loss')
        self.train_acc = metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss = metrics.Mean(name='test_loss')
        self.test_acc = metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(self, image, label):
        with tf.GradientTape() as tape:
            predictions = self.model(image) # propagation
            loss = self.loss_func(label, predictions) # calculate loss function
        gradients = tape.gradient(loss, self.model.trainable_variables) # calculate gradient
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))  #update parameters

        self.train_loss(loss)
        self.train_acc(label, predictions)

    @tf.function
    def test_step(self, image, label):
        predictions = self.model(image)
        t_loss = self.loss_func(label, predictions)

        self.test_loss(t_loss)
        self.test_acc(label, predictions)

    def train(self, epochs, training_images, train_labels, test_images, test_labels):
        template = 'Epoch {}, Loss: {:.5f}, Accuracy: {:.5f}, Test Loss: {:.5f}, Test Accuracy: {:.5f}, elapsed_time {:.5f}'

        for epoch in range(epochs):
            start = time.time()
            for image, label in zip(training_images, train_labels):
                self.train_step(image, label)
            elapsed_time = time.time() - start

            for test_image, test_label in zip(test_images, test_labels):
                self.test_step(test_image, test_label)

            print(template.format(epoch + 1,
                                  self.train_loss.result(),
                                  self.train_acc.result() * 100,
                                  self.test_loss.result(),
                                  self.test_acc.result() * 100,
                                  elapsed_time))