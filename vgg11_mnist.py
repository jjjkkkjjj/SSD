from models import VGG11
from train.trainer import Trainer
from datasets.mnist import data

if __name__ == '__main__':
    model = VGG11(10)
    model.build((None, 224, 224, 3))
    #model.summary()
    train_images, train_labels, test_images, test_labels = data()

    #Trainer(model=model, loss_func=)