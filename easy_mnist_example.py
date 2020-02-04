from model.core.model import Model
from model.core.architecture import *
from dataset.mnist import data
from model.dataset.dataset import DatasetClassification
from model.train.params import *

class MNIST(Model):
    _hidden_models = [
        Flatten('flatten1'),
        FullyConnection('fc1', outputnums=512, activationfunc='relu'),
        DropOut('do1', 0.2)

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
    iteration = IterationParams(epoch=5, batch_size=256)
    opt = OptimizationParams(learning_rate=10e-3, momentum=0.9)
    train_params = TrainingParams(loss, iteration, opt)

    dataset = DatasetClassification(10, train_images, train_labels, test_images, test_labels)

    model.train(dataset, train_params)
    # epoch: 5/5, loss: 1.513185, test accuracy: 0.95