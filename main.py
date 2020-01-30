from ssd.vgg16 import VGG16
from dataset.cifar10 import data

if __name__ == '__main__':
    vgg = VGG16(10)
    train_img, train_labels, test_img, test_labels = data()
    vgg.train(train_img, train_labels, test_img, test_labels, epoch=10, batch_size=32)