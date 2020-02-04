from .core.model import Model
from .core.architecture import *


class VGG16(Model):
    _hidden_models = [
        #Input('input', shape=[32, 32]),
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
        DropOut('do7', 0.5)
    ]
    def __init__(self, outputnum, *args, **kwargs):
        super().__init__(input_model=Input('input', rect=[28, 28], channel=1),
                         hidden_models=self._hidden_models,
                         output_model=FullyConnection('fc8', outputnums=outputnum, activationfunc='softmax'), *args, **kwargs)

        self.build()