from tensorflow.keras import Model, layers, activations

class VGG11(Model):
    def __init__(self, class_nums, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv1_1 = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='SAME', activation='relu')
        self.pool1 = layers.MaxPool2D((2, 2), strides=(2, 2), padding='SAME')

        self.conv2_1 = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='SAME', activation='relu')
        self.pool2 = layers.MaxPool2D((2, 2), strides=(2, 2), padding='SAME')

        self.conv3_1 = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='SAME', activation='relu')
        self.conv3_2 = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='SAME', activation='relu')
        self.pool3 = layers.MaxPool2D((2, 2), strides=(2, 2), padding='SAME')

        self.conv4_1 = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='SAME', activation='relu')
        self.conv4_2 = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='SAME', activation='relu')
        self.pool4 = layers.MaxPool2D((2, 2), strides=(2, 2), padding='SAME')

        self.conv5_1 = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='SAME', activation='relu')
        self.conv5_2 = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='SAME', activation='relu')
        self.pool5 = layers.MaxPool2D((2, 2), strides=(2, 2), padding='SAME')

        self.flatten6 = layers.Flatten()
        self.fc6 = layers.Dense(4096, activation='relu')
        self.do6 = layers.Dropout(rate=0.5)

        self.fc7 = layers.Dense(4096, activation='relu')
        self.do7 = layers.Dropout(rate=0.5)

        self.fc8 = layers.Dense(1000, activation='relu')

        self.fc9 = layers.Dense(class_nums, activation='softmax')


    def call(self, inputs, training=None, mask=None):
        x = self.conv1_1(inputs)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.pool4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.pool5(x)

        x = self.flatten6(x)
        x = self.fc6(x)
        x = self.do6(x)

        x = self.fc7(x)
        x = self.do7(x)

        x = self.fc8(x)

        return self.fc9(x)