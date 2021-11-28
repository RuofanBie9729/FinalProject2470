from tensorflow.keras.metrics import MeanIoU

import numpy as np
import tensorflow as tf

def IoU(labels, probs):
    """
    Calculates the model's prediction intersection over union by comparing
    probs to correct labels.

    :param probs: a matrix of size (num_inputs, 256, 256, 3); during training, this will be (batch_size, 256, 256, 3)
    containing the result of multiple convolution and feed forward layers
    :param labels: matrix of size (num_labels, 256, 256) containing the answers, during training, this will be (batch_size, 256, 256)
    :return: the IoU of the model as a Tensor
    """

    preds = tf.math.argmax(probs, axis=3)
    m = MeanIoU(num_classes=3)
    m.update_state(labels, preds)

    return m.result()


def pixel_acc(labels, probs):
    """
    Calculates the model's pixel accuracy by comparing probs to correct labels.

    :param probs: a matrix of size (num_inputs, 256, 256, 3); during training, this will be (batch_size, 256, 256, 3)
    containing the result of multiple convolution and feed forward layers
    :param labels: matrix of size (num_labels, 256, 256) containing the answers, during training, this will be (batch_size, 256, 256)
    :return: the pixel accuracy of the model as a float
    """

    labels = tf.cast(labels, tf.int32)
    labels = np.array(labels).flatten()

    preds = tf.math.argmax(probs, axis=3)
    preds = np.array(preds).flatten()

    return np.mean(labels == preds)


def mean_pixel_acc(labels, probs):
    """
    Calculates the model's mean pixel accuracy by comparing probs to correct labels.

    :param probs: a matrix of size (num_inputs, 256, 256, 3); during training, this will be (batch_size, 256, 256, 3)
    containing the result of multiple convolution and feed forward layers
    :param labels: matrix of size (num_labels, 256, 256) containing the answers, during training, this will be (batch_size, 256, 256)
    :return: the mean pixel accuracy of the model as a float
    """

    labels = tf.cast(labels, tf.int32)
    labels = np.array(labels).flatten()

    preds = tf.math.argmax(probs, axis=3)
    preds = np.array(preds).flatten()

    p0 = np.mean((labels == preds)[labels == 0])
    p1 = np.mean((labels == preds)[labels == 1])
    p2 = np.mean((labels == preds)[labels == 2])

    return np.mean([p0, p1, p2])


