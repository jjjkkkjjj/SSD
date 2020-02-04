from model.core.model import Model
from model.core.architecture import *
from dataset.mnist import data
from model.dataset.dataset import DatasetClassification
from model.train.params import *
from model.train.optimizer import *

"""
see https://www.tensorflow.org/tutorials/images/cnn?hl=ja
"""


class MNIST(Model):

    def __init__(self, *args, **kwargs):
        models = [
            Input('input', rect=[28, 28], channel=1),

            Convolution('conv1_1', kernel=[3, 3], kernelnums=32, strides=[1, 1]),
            MaxPooling('pool1', kernel=[2, 2], strides=[2, 2]),

            Convolution('conv2_1', kernel=[3, 3], kernelnums=64, strides=[1, 1]),
            MaxPooling('pool2', kernel=[2, 2], strides=[2, 2]),

            Convolution('conv3_1', kernel=[3, 3], kernelnums=64, strides=[1, 1]),
            Flatten('flatten1'),
            FullyConnection('fc4', outputnums=64, activationfunc='relu'),

            FullyConnection('output', outputnums=10, activationfunc='softmax')
        ]
        super().__init__(models, *args, **kwargs)



if __name__== '__main__':
    model = MNIST().build()

    train_images, train_labels, test_images, test_labels = data()

    loss = LossFunctionParams(func=LossFuncType.multinominal_logistic_regression, reg_type=LossRegularizationType.none)
    opt = OptimizationParams(optimizer=Adam(learning_rate=10e-3), epoch=5, batch_size=1000)
    train_params = TrainingParams(loss, opt)

    dataset = DatasetClassification(10, train_images, train_labels, test_images, test_labels)

    model.train(dataset, train_params)
    # epoch: 5/5, loss: 1.494311, test accuracy: 0.97