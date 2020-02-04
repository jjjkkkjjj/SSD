from model.core.model import Model
from model.core.architecture import *
from dataset.mnist import data
from model.dataset.dataset import DatasetClassification
from model.train.params import *

"""
see https://www.tensorflow.org/tutorials/images/cnn?hl=ja
"""


class MNIST(Model):
    _hidden_models = [
        Convolution('conv1_1', kernel=[3, 3], kernelnums=32, strides=[1, 1]),
        MaxPooling('pool1', kernel=[2, 2], strides=[2, 2]),

        Convolution('conv2_1', kernel=[3, 3], kernelnums=64, strides=[1, 1]),
        MaxPooling('pool2', kernel=[2, 2], strides=[2, 2]),

        Convolution('conv3_1', kernel=[3, 3], kernelnums=64, strides=[1, 1]),
        Flatten('flatten1'),
        FullyConnection('fc4', outputnums=64, activationfunc='relu'),

    ]
    def __init__(self, *args, **kwargs):
        super().__init__(input_model=Input('input', rect=[28, 28], channel=1),
                         hidden_models=self._hidden_models,
                         output_model=FullyConnection('output', outputnums=10, activationfunc='softmax'), *args, **kwargs)

        self.build()


if __name__== '__main__':
    model = MNIST()

    train_images, train_labels, test_images, test_labels = data()

    loss = LossFunctionParams(func=LossFuncType.multinominal_logistic_regression, reg_type=LossRegularizationType.none)
    iteration = IterationParams(epoch=5, batch_size=1000)
    opt = OptimizationParams(learning_rate=10e-3, momentum=0.9)
    train_params = TrainingParams(loss, iteration, opt)

    dataset = DatasetClassification(10, train_images, train_labels, test_images, test_labels)

    model.train(dataset, train_params)
    # epoch: 5/5, loss: 1.494311, test accuracy: 0.97