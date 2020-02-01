from ssd.vgg16 import VGG16
from dataset.cifar10 import data
from ssd.params.training import *

if __name__ == '__main__':
    vgg = VGG16(10, verbose=True)
    train_img, train_labels, test_img, test_labels = data()

    # training
    loss = LossFunction(func='multinominal_logistic_regression', reg_type='l2', decay=5*10e-4)
    iteration = Iteration(epoch=10, batch_size=32)
    opt = Optimization(learning_rate=10e-2, momentum=0.9)
    train_params = TrainingParams(loss, iteration, opt)

    vgg.train(train_img, train_labels, test_img, test_labels, params=train_params)