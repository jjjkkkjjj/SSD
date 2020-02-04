from model.core.model import Model
from model.core.architecture import *
from dataset.mnist import data
from model.dataset.dataset import DatasetClassification
from model.train.params import *
from model.train.optimizer import *

class MNIST(Model):

    def __init__(self, *args, **kwargs):
        models = [
            Input('input', rect=[28, 28], channel=1),
            Flatten('flatten1'),
            FullyConnection('fc1', outputnums=512, activationfunc='relu'),
            DropOut('do1', 0.2),
            FullyConnection('output', outputnums=10, activationfunc='softmax')
        ]
        super().__init__(models, *args, **kwargs)


def train():
    model = MNIST()

    train_images, train_labels, test_images, test_labels = data()

    loss = LossFunctionParams(func=LossFuncType.multinominal_logistic_regression, reg_type=LossRegularizationType.none)
    opt = OptimizationParams(optimizer=Adam(learning_rate=1e-3), epoch=5, batch_size=256)
    train_params = TrainingParams(loss, opt)

    dataset = DatasetClassification(10, train_images, train_labels, test_images, test_labels)

    model.train(dataset, train_params, savedir='./weights')
    # epoch: 5/5, loss: 1.513185, test accuracy: 0.95

def predict():
    model = MNIST().build()

    train_images, train_labels, test_images, test_labels = data()

    model.load('./weights/epoch5_loss1.5041388273239136_acc0.961899995803833')
    predicts = model.predict(test_images)
    print('acc: {0}%'.format((np.sum(predicts == test_labels) / len(test_labels))*100))
    # acc: 95.98%
if __name__== '__main__':
    #train()
    predict()