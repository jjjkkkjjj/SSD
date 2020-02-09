# this program was created by https://github.com/rykov8/ssd_keras

import numpy as np
import os
from xml.etree import ElementTree
import pickle

class XML_preprocessor(object):

    def __init__(self, data_path):
        self.path_prefix = data_path
        self.num_classes = 20
        self.data = dict()
        self._preprocess_XML()

    def _preprocess_XML(self):
        filenames = os.listdir(self.path_prefix)
        for filename in filenames:
            tree = ElementTree.parse(self.path_prefix + filename)
            root = tree.getroot()
            bounding_boxes = []
            #one_hot_classes = []
            class_numbers = []
            size_tree = root.find('size')
            width = float(size_tree.find('width').text)
            height = float(size_tree.find('height').text)
            for object_tree in root.findall('object'):
                for bounding_box in object_tree.iter('bndbox'):
                    xmin = float(bounding_box.find('xmin').text)/width
                    ymin = float(bounding_box.find('ymin').text)/height
                    xmax = float(bounding_box.find('xmax').text)/width
                    ymax = float(bounding_box.find('ymax').text)/height
                bounding_box = [xmin,ymin,xmax,ymax]
                bounding_boxes.append(bounding_box)
                class_name = object_tree.find('name').text
                class_number = self._to_number(class_name)
                class_numbers.append([class_number])
                #one_hot_class = self._to_one_hot(class_name)
                #one_hot_classes.append(one_hot_class)
            image_name = root.find('filename').text
            bounding_boxes = np.asarray(bounding_boxes)
            #one_hot_classes = np.asarray(one_hot_classes)
            class_numbers = np.asarray(class_numbers)
            #image_data = np.hstack((bounding_boxes, one_hot_classes))
            image_data = np.hstack([bounding_boxes, class_numbers])
            self.data[image_name] = image_data
    def _to_number(self, name):
        if name == 'aeroplane':
            return 0
        elif name == 'bicycle':
            return 1
        elif name == 'bird':
            return 2
        elif name == 'boat':
            return 3
        elif name == 'bottle':
            return 4
        elif name == 'bus':
            return 5
        elif name == 'car':
            return 6
        elif name == 'cat':
            return 7
        elif name == 'chair':
            return 8
        elif name == 'cow':
            return 9
        elif name == 'diningtable':
            return 10
        elif name == 'dog':
            return 11
        elif name == 'horse':
            return 12
        elif name == 'motorbike':
            return 13
        elif name == 'person':
            return 14
        elif name == 'pottedplant':
            return 15
        elif name == 'sheep':
            return 16
        elif name == 'sofa':
            return 17
        elif name == 'train':
            return 18
        elif name == 'tvmonitor':
            return 19
        else:
            print('unknown label: %s' %name)

    def _to_one_hot(self,name):
        one_hot_vector = [0] * self.num_classes
        if name == 'aeroplane':
            one_hot_vector[0] = 1
        elif name == 'bicycle':
            one_hot_vector[1] = 1
        elif name == 'bird':
            one_hot_vector[2] = 1
        elif name == 'boat':
            one_hot_vector[3] = 1
        elif name == 'bottle':
            one_hot_vector[4] = 1
        elif name == 'bus':
            one_hot_vector[5] = 1
        elif name == 'car':
            one_hot_vector[6] = 1
        elif name == 'cat':
            one_hot_vector[7] = 1
        elif name == 'chair':
            one_hot_vector[8] = 1
        elif name == 'cow':
            one_hot_vector[9] = 1
        elif name == 'diningtable':
            one_hot_vector[10] = 1
        elif name == 'dog':
            one_hot_vector[11] = 1
        elif name == 'horse':
            one_hot_vector[12] = 1
        elif name == 'motorbike':
            one_hot_vector[13] = 1
        elif name == 'person':
            one_hot_vector[14] = 1
        elif name == 'pottedplant':
            one_hot_vector[15] = 1
        elif name == 'sheep':
            one_hot_vector[16] = 1
        elif name == 'sofa':
            one_hot_vector[17] = 1
        elif name == 'train':
            one_hot_vector[18] = 1
        elif name == 'tvmonitor':
            one_hot_vector[19] = 1
        else:
            print('unknown label: %s' %name)

        return one_hot_vector

from scipy.misc import imread
from scipy.misc import imresize

def dump():
    # save annotation
    data = XML_preprocessor('dataset/voc/VOCdevkit/VOC2012/Annotations/').data
    #pickle.dump(data, open('dataset/voc/VOC2012-annotaion.pkl', 'wb'))

    images, boxes, labels = [], [], []
    for filename, value in data.items():
        img = imread('dataset/voc/VOCdevkit/VOC2012/JPEGImages/{0}'.format(filename)).astype('float32')
        # resize (annotation was normalized, therefore resizing will not influence annotation)
        img = imresize(img, (300, 300)).astype('float32')

        images.append(img)
        boxes.append(value[:, :4])
        labels.append(value[:, 4].astype('int'))
    images = np.asarray(images)
    boxes = np.asarray(boxes)
    labels = np.asarray(labels)

    np.save('dataset/voc/VOC2012-images.npy', images)
    np.save('dataset/voc/VOC2012-boxes.npy', boxes)
    np.save('dataset/voc/VOC2012-labels.npy', labels)


def data300():
    images = np.load('dataset/voc/VOC2012-images.npy', allow_pickle=True)
    boxes = np.load('dataset/voc/VOC2012-boxes.npy', allow_pickle=True)
    labels = np.load('dataset/voc/VOC2012-labels.npy', allow_pickle=True)

    # must be randomly split in dataset
    return images[:-2000], boxes[:-2000], labels[:-2000], images[-2000:], boxes[-2000:], labels[-2000:]
