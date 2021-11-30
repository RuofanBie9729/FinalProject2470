from tensorflow.keras.metrics import MeanIoU

import numpy as np
import tensorflow as tf
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os


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

    labels = tf.cast(labels, tf.int64)
    labels = tf.keras.layers.Flatten()(labels)

    preds = tf.math.argmax(probs, axis=3)
    preds = tf.keras.layers.Flatten()(preds)

    true_preds = tf.where(labels == preds, 1, 0)

    mask0 = tf.where(labels == 0, 1, 0)
    mask1 = tf.where(labels == 1, 1, 0)
    mask2 = tf.where(labels==2, 1, 0)

    p0 = tf.reduce_sum(tf.math.multiply(true_preds, mask0), axis=1) / tf.reduce_sum(mask0, axis=1)
    p0_not_nan = tf.dtypes.cast(tf.math.logical_not(tf.math.is_nan(p0)), dtype=tf.float64)
    p0 = tf.math.multiply_no_nan(p0, p0_not_nan)
    p0 = tf.reduce_mean(p0).numpy()

    p1 = tf.reduce_sum(tf.math.multiply(true_preds, mask1), axis=1) / tf.reduce_sum(mask1, axis=1)
    p1_not_nan = tf.dtypes.cast(tf.math.logical_not(tf.math.is_nan(p1)), dtype=tf.float64)
    p1 = tf.math.multiply_no_nan(p1, p1_not_nan)
    p1 = tf.reduce_mean(p1).numpy()

    p2 = tf.reduce_sum(tf.math.multiply(true_preds, mask2), axis=1) / tf.reduce_sum(mask2, axis=1)
    p2_not_nan = tf.dtypes.cast(tf.math.logical_not(tf.math.is_nan(p2)), dtype=tf.float64)
    p2 = tf.math.multiply_no_nan(p2, p2_not_nan)
    p2 = tf.reduce_mean(p2).numpy()

    return np.mean([p0, p1, p2])


def show_seg(model, inputs, labels, model_name):
    num_obs = inputs.shape[0]

    probs = model(inputs)
    preds = tf.math.argmax(probs, axis=3)
    preds = tf.cast(preds, tf.double)

    labels = tf.cast(labels, tf.double)

    images = tf.concat([tf.reshape(inputs, (num_obs, 256, 256)), labels, preds], axis=0)

    fig = plt.figure(figsize=(3 * num_obs, num_obs))
    gspec = gridspec.GridSpec(3, num_obs)
    gspec.update(wspace=0.05, hspace=0.05)

    for i in range(3 * num_obs):
        ax = plt.subplot(gspec[i])
        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        plt.imshow(images[i])

    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", model_name + "_seg_results.pdf")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)