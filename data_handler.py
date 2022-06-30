from functools import partial
from matplotlib import image
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds

def _normalize(data, labels, classes):
    data = tf.cast(data, tf.float32)
    data = tf.divide(data, 255.0)
    labels = tf.one_hot(labels, classes)
    return data, labels

def cifar10_dataset():
    dataset = tfds.load("cifar10", split="train", as_supervised=True)
    dataset = dataset.map(partial(_normalize, classes=10)).cache()
    return dataset
    
def cifar100_dataset():
    dataset = tfds.load("cifar100", split="train", as_supervised=True)
    dataset = dataset.map(partial(_normalize, classes=100)).cache()
    return dataset

"""
def imagenet_dataset():
    dataset = tfds.load("imagenet2012", split="train", as_supervised=True)
    dataset = dataset.map(partial(_normalize, classes=1000)).cache()
    return dataset
"""


if __name__ == '__main__':
    #imagenet()

    data = cifar10_dataset()

    b_data = data.batch(32)

    im, lab = next(iter(b_data))


    #print(x.shape)

