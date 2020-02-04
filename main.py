from ssd.vgg16 import VGG16
from dataset.mnist import data
from ssd.params.training import *
from ssd.data.dataset import DatasetClassification

if __name__ == '__main__':
    vgg = VGG16(10, verbose=True)
    train_img, train_labels, test_img, test_labels = data()

    # training
    loss = LossFunctionParams(func=LossFuncType.square_error, reg_type=LossRegularizationType.none, decay=5 * 10e-4)
    iteration = IterationParams(epoch=5, batch_size=1000)
    opt = OptimizationParams(learning_rate=10e-3, momentum=0.9)
    train_params = TrainingParams(loss, iteration, opt)

    dataset = DatasetClassification(10, train_img, train_labels, test_img, test_labels)
    vgg.train(dataset, params=train_params)