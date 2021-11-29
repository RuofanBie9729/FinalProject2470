from preprocessing import get_data
from metrics import IoU, pixel_acc, mean_pixel_acc, show_seg
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

import tensorflow as tf
import numpy as np
import math


def maxpool2d_with_argmax(incoming, pool_size=2, stride=2, name='maxpool_with_argmax'):

    x = incoming
    filter_shape = [1, pool_size, pool_size, 1]
    strides = [1, stride, stride, 1]

    with tf.name_scope(name):
        _, mask = tf.nn.max_pool_with_argmax(x, ksize=filter_shape, strides=strides, padding='SAME')
        mask = tf.stop_gradient(mask)

        pooled = tf.nn.max_pool(x, ksize=filter_shape, strides=strides, padding='SAME')

    return pooled, mask


def maxunpool2d(incoming, mask, stride=2, name='unpool'):

    x = incoming

    input_shape = incoming.get_shape().as_list()
    strides = [1, stride, stride, 1]
    output_shape = (input_shape[0], input_shape[1] * strides[1], input_shape[2] * strides[2], input_shape[3])

    flat_output_shape = [output_shape[0], np.prod(output_shape[1:])]

    with tf.name_scope(name):
        flat_input_size = tf.size(x)
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=mask.dtype),
                                 shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(mask) * batch_range
        b = tf.reshape(b, [flat_input_size, 1])
        mask_ = tf.reshape(mask, [flat_input_size, 1])
        mask_ = tf.concat([b, mask_], 1)

        x_ = tf.reshape(x, [flat_input_size])
        ret = tf.scatter_nd(mask_, x_, shape=flat_output_shape)
        ret = tf.reshape(ret, output_shape)

    return ret


class Model(tf.keras.Model):
    def __init__(self):

        super(Model, self).__init__()

        self.batch_size = 16
        self.optimizer = Adam(learning_rate=1e-5)
        self.loss_list = []

    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.

        :param inputs: images, shape of (num_inputs, 256, 256, 1); during training, the shape is (batch_size, 256, 256, 1)
        :return: probs - a matrix of shape (num_inputs, 256, 256, 3); during training, it would be (batch_size, 256, 256, 3)
        """

        conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1, ind1 = maxpool2d_with_argmax(conv1, 2, 2)

        conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2, ind2 = maxpool2d_with_argmax(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
        pool3, ind3 = maxpool2d_with_argmax(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
        pool4, ind4 = maxpool2d_with_argmax(conv4)

        conv5 = Conv2D(512, 3, activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, 3, activation='relu', padding='same')(conv5)
        conv5 = Conv2D(512, 3, activation='relu', padding='same')(conv5)
        pool5, ind5 = maxpool2d_with_argmax(conv5)

        fc6 = Conv2D(4096, 8, activation='relu')(pool5)
        fc7 = Conv2D(4096, 1, activation='relu')(fc6)

        deconv8 = Conv2DTranspose(512, 8, activation='relu')(fc7)
        unpool8 = maxunpool2d(deconv8, ind5)

        deconv9 = Conv2DTranspose(512, 3, activation='relu', padding='same')(unpool8)
        deconv9 = Conv2DTranspose(512, 3, activation='relu', padding='same')(deconv9)
        deconv9 = Conv2DTranspose(512, 3, activation='relu', padding='same')(deconv9)
        unpool9 = maxunpool2d(deconv9, ind4)

        deconv10 = Conv2DTranspose(512, 3, activation='relu', padding='same')(unpool9)
        deconv10 = Conv2DTranspose(512, 3, activation='relu', padding='same')(deconv10)
        deconv10 = Conv2DTranspose(256, 3, activation='relu', padding='same')(deconv10)
        unpool10 = maxunpool2d(deconv10, ind3)

        deconv11 = Conv2DTranspose(256, 3, activation='relu', padding='same')(unpool10)
        deconv11 = Conv2DTranspose(256, 3, activation='relu', padding='same')(deconv11)
        deconv11 = Conv2DTranspose(128, 3, activation='relu', padding='same')(deconv11)
        unpool11 = maxunpool2d(deconv11, ind2)

        deconv12 = Conv2DTranspose(128, 3, activation='relu', padding='same')(unpool11)
        deconv12 = Conv2DTranspose(64, 3, activation='relu', padding='same')(deconv12)
        unpool12 = maxunpool2d(deconv12, ind1)

        deconv13 = Conv2DTranspose(64, 3, activation='relu', padding='same')(unpool12)
        deconv13 = Conv2DTranspose(64, 3, activation='relu', padding='same')(deconv13)

        conv14 = Conv2D(3, 1, activation='softmax', padding='same')(deconv13)

        return conv14

    def loss(self, probs, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.

        :param probs: a matrix of shape (batch_size, 256, 256, 3)
        :param labels: during training, matrix of shape (batch_size, 256, 256) containing the train labels
        :return: the loss of the model as a Tensor
        """

        loss = SparseCategoricalCrossentropy()(labels, probs)

        return tf.reduce_mean(loss)


def train(model, train_inputs, train_labels):
    """
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training),
    shape (num_inputs, 256, 256, 1)
    :param train_labels: train labels (all labels to use for training),
    shape (num_labels, 256, 256)
    :return: list of losses per batch
    """

    batch_size = model.batch_size
    n_batch = math.ceil(len(train_inputs) / batch_size)

    model.loss_list = []

    for i in range(n_batch):
        starting_index = i * batch_size
        batch_inputs = train_inputs[starting_index:starting_index + batch_size]
        batch_labels = train_labels[starting_index:starting_index + batch_size]

        with tf.GradientTape() as tape:
            probs = model(batch_inputs)
            loss = model.loss(probs, batch_labels)
            model.loss_list.append(loss.numpy())

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return None


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels.

    :param test_inputs: test data (all images to be tested),
    shape (num_inputs, 256, 256, 1)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, 256, 256)
    :return: IoU, pixel accuracy and mean pixel accuracy
    """

    probs = model(test_inputs)
    iou = IoU(test_labels, probs)
    acc = pixel_acc(test_labels, probs)
    mean_acc = mean_pixel_acc(test_labels, probs)

    return iou, acc, mean_acc


def main():

    train_inputs, train_labels = get_data('data/train_img.npy', 'data/train_lab.npy', aug='both')
    train_inputs = tf.reshape(train_inputs, (train_inputs.shape[0], 256, 256, 1))
    test_inputs, test_labels = get_data('data/test_img.npy', 'data/test_lab.npy')
    test_inputs = tf.reshape(test_inputs, (test_inputs.shape[0], 256, 256, 1))

    # create model
    model = Model()

    # train
    for i in range(1):
        train(model, train_inputs[:16], train_labels[:16])
        print(f"Train Epoch: {i} \tLoss: {np.mean(model.loss_list):.6f}")
        iou, acc, mean_acc = test(model, test_inputs[:16], test_labels[:16])
        print(f"--IoU: {iou:.6f}  --pixel accuracy: {acc:.6f}  --mean pixel accuracy: {mean_acc:.6f}")

    show_seg(model, test_inputs[:10], test_labels[:10])


if __name__ == '__main__':
    main()
