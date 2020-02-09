from .core.model import ModelObjectDetection
from .core.opt import ObjectDetectionMixin
from .vgg16 import VGG16
from .train.params import TrainingParams
from .core.layer_models import *
from .dataset.dataset import *
from .ssd.loss import get_loss

import logging
import tensorflow as tf

class SSD300(ModelObjectDetection, ObjectDetectionMixin):
    #https://github.com/rykov8/ssd_keras/blob/master/ssd.py
    def __init__(self, class_num, *args, **kwargs):
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
        self.class_num = class_num
        feature_kernelnums = 4 * (self.class_num + 4)
        # store feature layers and models
        self.extra_feature_models = [
            Convolution('conv4_3_featuremap', kernel=[3, 3], kernelnums=feature_kernelnums, strides=[1, 1], padding='SAME'),

            Convolution('conv7_featuremap', kernel=[3, 3], kernelnums=feature_kernelnums, strides=[1, 1], padding='SAME'),

            Convolution('conv8_2_featuremap', kernel=[3, 3], kernelnums=feature_kernelnums, strides=[1, 1], padding='SAME'),

            Convolution('conv9_2_featuremap', kernel=[3, 3], kernelnums=feature_kernelnums, strides=[1, 1], padding='SAME'),

            Convolution('conv10_2_featuremap', kernel=[3, 3], kernelnums=feature_kernelnums, strides=[1, 1], padding='SAME'),

            Convolution('conv11_2_featuremap', kernel=[3, 3], kernelnums=feature_kernelnums, strides=[1, 1], padding='SAME')
        ]

        self.featuremap_boxes = []
        self.featuremap_ratios = []
        self.featuremaps_layers = []
        self.confidences =[]
        self.locations = []

    def build(self):
        tmp = self.verbose
        self.verbose = False
        super().build()
        self.verbose = tmp

        # build feature layer models
        self.featuremap_info = [4, 6, 6, 6, 6, 6]
        self.featuremap_ratios = [
            [1.0, 1.0, 2.0, 1.0 / 2.0],
            [1.0, 1.0, 2.0, 1.0 / 2.0, 3.0, 1.0 / 3.0],
            [1.0, 1.0, 2.0, 1.0 / 2.0, 3.0, 1.0 / 3.0],
            [1.0, 1.0, 2.0, 1.0 / 2.0, 3.0, 1.0 / 3.0],
            [1.0, 1.0, 2.0, 1.0 / 2.0, 3.0, 1.0 / 3.0],
            [1.0, 1.0, 2.0, 1.0 / 2.0, 3.0, 1.0 / 3.0],
        ]
        extra_feature_input = [self.conv4_3, self.conv7, self.conv8_2, self.conv9_2, self.conv10_2, self.conv11_2]
        for i in range(len(extra_feature_input)):
            input = extra_feature_input[i]
            layer_model = self.extra_feature_models[i]
            boxes = self.featuremap_boxes[i]

            layer = self.get_layer(input, layer_model)
            self.featuremaps_layers.append(layer)
            self.set_layer_attribute(layer, layer_model)

            # get confidence and location
            shape = layer.get_shape().as_list()
            height, width = shape[1], shape[2]
            output = tf.reshape(layer, [-1, width*height*boxes, self.class_num + 4])

            confidence = output[:, :, :self.class_num]
            location = output[:, :, self.class_num:]
            self.set_layer_attribute_name(confidence, 'confidence{0}'.format(i + 1))
            self.set_layer_attribute_name(location, 'location{0}'.format(i + 1))

            self.confidences.append(confidence)
            self.locations.append(location)

        session = tf.compat.v1.Session()
        with tf.name_scope('summary'):
            merged = tf.compat.v1.summary.merge_all()
            writer = tf.compat.v1.summary.FileWriter('./logs', session.graph)

        logging.debug("\nBuilding model was succeeded.\n")

        return self

    @property
    def isBuilt(self):
        return len(self.featuremaps_layers) > 0

    def score(self):
        if not self.isBuilt:
            message = 'score function must be called after calling build function'
            raise UnBuiltError(message)
        return self.confidences, self.locations

    
    def train(self, dataset, params, savedir=None):
        dataset = check_type(dataset, 'dataset', DatasetObjectDetection, self, funcnames='train')
        
        dataset: DatasetObjectDetection
        # get params for training
        self.params = check_type(params, 'params', TrainingParams, self, funcnames='train')

        loss_params = self.params.lossfunc_params
        opt_params = self.params.opt_params

        input = self.input_layer

        #gt_labels =

        loss = get_loss()



    def predict(self, X):
        pass