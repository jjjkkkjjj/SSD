from ssd.base.model import Model
from ssd.base.architecture import *
from dataset.mnist import data
from ssd.data.dataset import DatasetClassification
from ssd.params.training import *

class MNIST(Model):
    _hidden_models = [
        Convolution('conv1_1', kernel=[3, 3], kernelnums=32, strides=[1, 1]),
        Convolution('conv1_2', kernel=[3, 3], kernelnums=32, strides=[1, 1]),
        MaxPooling('pool1', kernel=[2, 2], strides=[2, 2]),

        Convolution('conv2_1', kernel=[3, 3], kernelnums=64, strides=[1, 1]),
        Convolution('conv2_2', kernel=[3, 3], kernelnums=64, strides=[1, 1]),
        MaxPooling('pool2', kernel=[2, 2], strides=[2, 2]),

        Convolution('conv3_1', kernel=[3, 3], kernelnums=128, strides=[1, 1]),
        Convolution('conv3_2', kernel=[3, 3], kernelnums=128, strides=[1, 1]),
        Convolution('conv3_3', kernel=[3, 3], kernelnums=128, strides=[1, 1]),
        MaxPooling('pool3', kernel=[2, 2], strides=[2, 2]),

        Flatten('flatten1'),
        FullyConnection('fc4', outputnums=4096, activationfunc='relu'),
        DropOut('do4', rate=0.5),

    ]
    def __init__(self, *args, **kwargs):
        super().__init__(input_model=Input('input', rect=[28, 28], channel=1),
                         hidden_models=self._hidden_models,
                         output_model=FullyConnection('fc5', outputnums=10, activationfunc='relu'), *args, **kwargs)

        self.build()


if __name__== '__main__':
    model = MNIST()

    train_images, train_labels, test_images, test_labels = data()

    loss = LossFunctionParams(func=LossFuncType.square_error, reg_type=LossRegularizationType.l1,
                              decay=5 * 10e-4)
    iteration = IterationParams(epoch=10, batch_size=256)
    opt = OptimizationParams(learning_rate=10e-2, momentum=0.9)
    train_params = TrainingParams(loss, iteration, opt)

    dataset = DatasetClassification(10, train_images, train_labels, test_images, test_labels)

    model.train(dataset, train_params)