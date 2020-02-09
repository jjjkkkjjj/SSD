from model.ssd300 import SSD300
from dataset.voc import data300, dump
from model.train.params import *
from model.dataset.dataset import DatasetObjectDetection
from model.train.optimizer import *

if __name__ == '__main__':
    #ssd = SSD300(class_num=10, verbose=True).build()
    #dump()
    train_images, train_boxes, train_labels, test_images, test_boxes, test_labels = data300()

    # training
    #loss = LossFunctionParams(func=LossFuncType.square_error, reg_type=LossRegularizationType.none, decay=5 * 10e-4)
    #opt = OptimizationParams(optimizer=Adam(10e-3), epoch=5, batch_size=1000)
    #train_params = TrainingParams(loss, opt)

    dataset = DatasetObjectDetection(20, train_images, train_boxes, train_labels, test_images, test_boxes, test_labels)
    #vgg.train(dataset, params=train_params)