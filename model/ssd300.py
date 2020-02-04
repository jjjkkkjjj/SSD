from .core.model import Model
from .vgg16 import VGG16
from .core.architecture import *

class SSD300(Model):
    _hidden_models = [].extend(VGG16._hidden_models[:17]).extend(
        [
            Convolution('conv6_1', kernel=[3, 3], kernelnums=1024, strides=[1, 1], padding='SAME'),
            Convolution('conv_feature_1', kernel=[1, 1], kernelnums=256, strides=[2, 2], padding='SAME'),
            Convolution('conv_feature_2', kernel=[1, 1], kernelnums=128, strides=[2, 2], padding='SAME'),
            Convolution('conv_feature_3', kernel=[1, 1], kernelnums=128, strides=[1, 1], padding='VALID'),
            Convolution('conv_feature_4', kernel=[1, 1], kernelnums=128, strides=[1, 1], padding='VALID')
        ]
    )
