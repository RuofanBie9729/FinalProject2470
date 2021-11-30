from __future__ import absolute_import
from metrics import IoU, pixel_acc, mean_pixel_acc, show_seg
from tensorflow.keras.metrics import MeanIoU

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, Conv2DTranspose
from tensorflow.math import exp, sqrt, square
import numpy as np
import random
import math


class FCN(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for our FCN that
        implements image segmentation.
        """
        super(FCN, self).__init__()

        self.batch_size = 1
        # Initialize convolutional layers and deconvolutional layer
        self.convnets = tf.keras.Sequential()
        self.deconv = tf.keras.Sequential()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

        self.convnets.add(Conv2D(2, 3, activation='relu'))
        self.convnets.add(Conv2D(2, 3, activation='relu'))
        self.convnets.add(Conv2D(4, 3, activation='relu'))
        self.convnets.add(Conv2D(4, 3, activation='relu'))
        # self.convnets.add(tf.keras.layers.MaxPooling2D())
        self.convnets.add(Conv2D(8, 3, activation='relu'))
        self.convnets.add(Conv2D(8, 3, activation='relu'))
        self.convnets.add(Conv2D(8, 3, activation='relu'))
        self.convnets.add(Conv2D(16, 3, activation='relu'))
        self.convnets.add(Conv2D(16, 1, activation='relu'))
        self.deconv.add(Conv2DTranspose(3, 17))
        self.deconv.add(Conv2D(3, 1, activation='softmax', padding='same'))

    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.

        :param inputs: images, shape of (num_inputs, 256, 256, 1)
        :return: logits - Flatten predictied image, a matrix of shape (num_inputs, 256*256)
        """
        FCNOutput = self.convnets(inputs)
        prbs = self.deconv(FCNOutput)

        return prbs

    def loss(self, prbs, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.

        :param logits: during training, a matrix of shape (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """

        labels = tf.cast(labels, tf.int32)

        class_weights = tf.constant([1, 3, 3])
        class_weights = class_weights / tf.reduce_sum(class_weights)
        sample_weights = tf.gather(class_weights, indices=tf.cast(labels, tf.int32))

        loss = SparseCategoricalCrossentropy()(labels, prbs, sample_weights)

        return tf.reduce_mean(loss)


def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs
    and labels - ensure that they are shuffled in the same order using tf.gather or zipping.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training),
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training),
    shape (num_labels, num_classes)
    :return: Optionally list of losses per batch to use for visualize_loss
    '''

    batch_size = model.batch_size
    n_batch = math.ceil(len(train_inputs) / batch_size)

    loss_list = []

    for i in range(n_batch):
        starting_index = i * batch_size
        batch_inputs = train_inputs[starting_index:starting_index + batch_size]
        batch_labels = train_labels[starting_index:starting_index + batch_size]

        with tf.GradientTape() as tape:
            probs = model(batch_inputs)
            loss = model.loss(probs, batch_labels)
            print(loss)
            loss_list.append(loss.numpy())

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss_list


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels.
    :param test_inputs: test data (all images to be tested),
    shape (num_inputs, 256, 256, 1)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, 256, 256)
    :return: IoU, pixel accuracy and mean pixel accuracy
    """

    batch_size = model.batch_size
    n_batch = math.ceil(len(test_inputs) / batch_size)

    iou = 0
    acc = 0
    mean_acc = 0

    for i in range(n_batch):
        starting_index = i * batch_size
        batch_inputs = test_inputs[starting_index:starting_index + batch_size]
        batch_labels = test_labels[starting_index:starting_index + batch_size]

        probs = model(batch_inputs)
        iou += IoU(batch_labels, probs).numpy() * len(batch_inputs)
        acc += pixel_acc(batch_labels, probs) * len(batch_inputs)
        mean_acc += mean_pixel_acc(batch_labels, probs) * len(batch_inputs)

    return iou / len(test_inputs), acc / len(test_inputs), mean_acc / len(test_inputs)


def main():
    train_inputs, train_labels = get_data('../data/train_img_r.npy', '../data/train_lab_r.npy', 'both')
    train_inputs = tf.reshape(train_inputs, (train_inputs.shape[0], 256, 256, 1))
    test_inputs, test_labels = get_data('../data/test_img_r.npy', '../data/test_lab_r.npy')
    test_inputs = tf.reshape(test_inputs, (test_inputs.shape[0], 256, 256, 1))

    # create model
    model = Model()

    # train
    for i in range(50):
        train(model, train_inputs, train_labels)
        print(f"Train Epoch: {i}  Loss: {np.mean(model.loss_list):.6f}")
        iou, acc, mean_acc = test(model, test_inputs, test_labels)
        print(f"--IoU: {iou:.6f}  --pixel accuracy: {acc:.6f}  --mean pixel accuracy: {mean_acc:.6f}")

        if (i + 1) % 10 == 0:
            show_seg(model, test_inputs[:10], test_labels[:10], 'fcn_both' + str(i + 1))


if __name__ == '__main__':
    main()
