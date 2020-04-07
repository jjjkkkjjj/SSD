from tensorflow.keras import layers, activations
from .vgg11 import VGG11

class VGG16(VGG11):
    def __init__(self, class_num, *args, **kwargs):
        super().__init__(class_num, *args, **kwargs)

        self.conv1_2 = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='SAME', activation='relu')

        self.conv2_2 = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='SAME', activation='relu')

        self.conv3_3 = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='SAME', activation='relu')

        self.conv4_3 = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='SAME', activation='relu')

        self.conv5_3 = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='SAME', activation='relu')

    def call(self, inputs, training=None, mask=None):
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pool4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)

        x = self.flatten6(x)
        x = self.fc6(x)
        x = self.do6(x)

        x = self.fc7(x)
        x = self.do7(x)

        x = self.fc8(x)

        return self.fc9(x)