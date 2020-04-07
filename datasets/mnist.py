from tensorflow.keras import datasets

def data():
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    return train_images.reshape(-1, 28, 28, 1), train_labels, test_images.reshape(-1, 28, 28, 1), test_labels
