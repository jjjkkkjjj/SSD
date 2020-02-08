from .core.model import Model
from .core.architecture import *


class VGG11(Model):
    def __init__(self, outputnum, input_rect=(224, 224), input_channel=3, *args, **kwargs):
        models = [
            Input('input', rect=input_rect, channel=input_channel),
            Convolution('conv1_1', kernel=[3, 3], kernelnums=64, strides=[1, 1], padding='SAME'),
            MaxPooling('pool1', kernel=[2, 2], strides=[2, 2], padding='SAME'),

            Convolution('conv2_1', kernel=[3, 3], kernelnums=128, strides=[1, 1], padding='SAME'),
            MaxPooling('pool2', kernel=[2, 2], strides=[2, 2], padding='SAME'),

            Convolution('conv3_1', kernel=[3, 3], kernelnums=256, strides=[1, 1], padding='SAME'),
            Convolution('conv3_2', kernel=[3, 3], kernelnums=256, strides=[1, 1], padding='SAME'),
            MaxPooling('pool3', kernel=[2, 2], strides=[2, 2], padding='SAME'),

            Convolution('conv4_1', kernel=[3, 3], kernelnums=512, strides=[1, 1], padding='SAME'),
            Convolution('conv4_2', kernel=[3, 3], kernelnums=512, strides=[1, 1], padding='SAME'),
            MaxPooling('pool4', kernel=[2, 2], strides=[2, 2], padding='SAME'),

            Convolution('conv5_1', kernel=[3, 3], kernelnums=512, strides=[1, 1], padding='SAME'),
            Convolution('conv5_2', kernel=[3, 3], kernelnums=512, strides=[1, 1], padding='SAME'),
            MaxPooling('pool5', kernel=[2, 2], strides=[2, 2], padding='SAME'),

            Flatten('flatten1'),
            FullyConnection('fc6', outputnums=4096, activationfunc='relu'),
            DropOut('do6', rate=0.5),

            FullyConnection('fc7', outputnums=4096, activationfunc='relu'),
            DropOut('do7', 0.5),

            FullyConnection('fc8', outputnums=outputnum, activationfunc='relu')
        ]
        super().__init__(models, *args, **kwargs)
