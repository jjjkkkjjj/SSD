from model.core.model import Model
from model.core.architecture import *
from dataset.mnist import data
from model.dataset.dataset import DatasetClassification
from model.train.params import *
from model.train.optimizer import *

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
    opt = OptimizationParams(optimizer=Adam(learning_rate=1e-3), epoch=5, batch_size=256)
    train_params = TrainingParams(loss, opt)

    dataset = DatasetClassification(10, train_images, train_labels, test_images, test_labels)

    model.train(dataset, train_params, savedir='./weights')
    # epoch: 5/5, loss: 1.513185, test accuracy: 0.95