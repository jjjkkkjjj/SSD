from model.ssd300 import SSD300
from dataset.mnist import data
from model.train.params import *
from model.dataset.dataset import DatasetClassification
from model.train.optimizer import *

if __name__ == '__main__':
    ssd = SSD300(outputnum=10, verbose=True).build()
    """
    train_img, train_labels, test_img, test_labels = data()

    # training
    loss = LossFunctionParams(func=LossFuncType.square_error, reg_type=LossRegularizationType.none, decay=5 * 10e-4)
    opt = OptimizationParams(optimizer=Adam(10e-3), epoch=5, batch_size=1000)
    train_params = TrainingParams(loss, opt)

    dataset = DatasetClassification(10, train_img, train_labels, test_img, test_labels)
    vgg.train(dataset, params=train_params)
    """