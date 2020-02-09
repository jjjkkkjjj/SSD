from .core.model import ModelObjectDetection
from .core.opt import ObjectDetectionMixin
from .vgg16 import VGG16
from .core.layer_models import *
from .dataset.dataset import *

import logging

class SSD300(ModelObjectDetection, ObjectDetectionMixin):
    #https://github.com/rykov8/ssd_keras/blob/master/ssd.py
    def __init__(self, outputnum, *args, **kwargs):
        #outputnum=10 is dummy
        vgg16 = VGG16(outputnum=10)
        models = vgg16.all_models[:18]

        models.extend([
            MaxPooling('pool5', kernel=[3, 3], strides=[1, 1], padding='SAME'),

            AtrousConvolution('conv6', dilation_rate=6, kernel=[3, 3], kernelnums=1024, strides=[1, 1], padding='SAME'),

            Convolution('conv7', kernel=[1, 1], kernelnums=1024, strides=[1, 1], padding='SAME'),

            Convolution('conv8_1', kernel=[1, 1], kernelnums=256, strides=[1, 1], padding='SAME'),
            Convolution('conv8_2', kernel=[3, 3], kernelnums=512, strides=[2, 2], padding='SAME'),

            Convolution('conv9_1', kernel=[1, 1], kernelnums=128, strides=[1, 1], padding='SAME'),
            Convolution('conv9_2', kernel=[3, 3], kernelnums=256, strides=[2, 2], padding='SAME'),

            Convolution('conv10_1', kernel=[1, 1], kernelnums=128, strides=[1, 1], padding='SAME'),
            Convolution('conv10_2', kernel=[3, 3], kernelnums=256, strides=[1, 1], padding='SAME'),

            Convolution('conv11_1', kernel=[1, 1], kernelnums=128, strides=[1, 1], padding='SAME'),
            Convolution('conv11_2', kernel=[3, 3], kernelnums=256, strides=[1, 1], padding='SAME')
        ])

        super().__init__(models, *args, **kwargs)
        feature_kernelnums = 4 * (outputnum + 4)
        # store feature layers and models
        self.extra_feature_models = [
            Convolution('conv4_3_feature', kernel=[3, 3], kernelnums=feature_kernelnums, strides=[1, 1], padding='SAME'),

            Convolution('conv7_feature', kernel=[3, 3], kernelnums=feature_kernelnums, strides=[1, 1], padding='SAME'),

            Convolution('conv8_2_feature', kernel=[3, 3], kernelnums=feature_kernelnums, strides=[1, 1], padding='SAME'),

            Convolution('conv9_2_feature', kernel=[3, 3], kernelnums=feature_kernelnums, strides=[1, 1], padding='SAME'),

            Convolution('conv10_2_feature', kernel=[3, 3], kernelnums=feature_kernelnums, strides=[1, 1], padding='SAME'),

            Convolution('conv11_2_feature', kernel=[3, 3], kernelnums=feature_kernelnums, strides=[1, 1], padding='SAME')
        ]
        self.feature_layers = []

    def build(self):
        tmp = self.verbose
        self.verbose = False
        super().build()
        self.verbose = tmp

        # build feature layer models
        extra_feature_input = [self.conv4_3, self.conv7, self.conv8_2, self.conv9_2, self.conv10_2, self.conv11_2]
        for input, layer_model in zip(extra_feature_input, self.extra_feature_models):
            layer = self.get_layer(input, layer_model)
            self.feature_layers.append(layer)
            self.set_layer_attribute(layer, layer_model)

        logging.debug("\nBuilding model was succeeded.\n")

        return self
    
    def train(self, dataset, params, savedir=None):
        dataset = check_type(dataset, 'dataset', DatasetObjectDetection, self, funcnames='train')
        
        dataset: DatasetObjectDetection