from .core.model import ModelClassifier
from .core.opt import ClassifierMixin
from .core.layer_models import *


class VGG16(ModelClassifier, ClassifierMixin):
    def __init__(self, outputnum, *args, **kwargs):
        models = [
            Input('input', rect=[32, 32], channel=3),
            Convolution('conv1_1', kernel=[3, 3], kernelnums=64, strides=[1, 1], padding='SAME'),
            Convolution('conv1_2', kernel=[3, 3], kernelnums=64, strides=[1, 1], padding='SAME'),
            MaxPooling('pool1', kernel=[2, 2], strides=[2, 2], padding='SAME'),

            Convolution('conv2_1', kernel=[3, 3], kernelnums=128, strides=[1, 1], padding='SAME'),
            Convolution('conv2_2', kernel=[3, 3], kernelnums=128, strides=[1, 1], padding='SAME'),
            MaxPooling('pool2', kernel=[2, 2], strides=[2, 2], padding='SAME'),

            Convolution('conv3_1', kernel=[3, 3], kernelnums=256, strides=[1, 1], padding='SAME'),
            Convolution('conv3_2', kernel=[3, 3], kernelnums=256, strides=[1, 1], padding='SAME'),
            Convolution('conv3_3', kernel=[3, 3], kernelnums=256, strides=[1, 1], padding='SAME'),
            MaxPooling('pool3', kernel=[2, 2], strides=[2, 2], padding='SAME'),

            Convolution('conv4_1', kernel=[3, 3], kernelnums=512, strides=[1, 1], padding='SAME'),
            Convolution('conv4_2', kernel=[3, 3], kernelnums=512, strides=[1, 1], padding='SAME'),
            Convolution('conv4_3', kernel=[3, 3], kernelnums=512, strides=[1, 1], padding='SAME'),
            MaxPooling('pool4', kernel=[2, 2], strides=[2, 2], padding='SAME'),

            Convolution('conv5_1', kernel=[3, 3], kernelnums=512, strides=[1, 1], padding='SAME'),
            Convolution('conv5_2', kernel=[3, 3], kernelnums=512, strides=[1, 1], padding='SAME'),
            Convolution('conv5_3', kernel=[3, 3], kernelnums=512, strides=[1, 1], padding='SAME'),
            MaxPooling('pool5', kernel=[2, 2], strides=[2, 2], padding='SAME'),

            Flatten('flatten1'),
            FullyConnection('fc6', outputnums=4096, activationfunc='relu'),
            DropOut('do6', rate=0.5),

            FullyConnection('fc7', outputnums=4096, activationfunc='relu'),
            DropOut('do7', 0.5),

            FullyConnection('fc8', outputnums=outputnum, activationfunc='relu')
        ]
        super().__init__(models, *args, **kwargs)
