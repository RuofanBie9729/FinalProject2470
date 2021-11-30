from preprocessing import get_data
from metrics import IoU, pixel_acc, mean_pixel_acc, show_seg
from tensorflow.keras import Sequential
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

        self.batch_size = 1
        self.optimizer = Adam(learning_rate=1e-5)
        self.loss_list = []

        self.conv1 = self.conv_block(16)
        self.conv2 = self.conv_block(32)
        self.conv3 = self.conv_block(64, layers=3)

        self.connect = Sequential()
        self.connect.add(Conv2D(512, 32, activation='relu'))
        self.connect.add(Conv2D(512, 1, activation='relu'))
        self.connect.add(Conv2DTranspose(64, 32, activation='relu'))

        self.deconv1 = self.deconv_block(64, layers=3)
        self.deconv2 = self.deconv_block(32, layers=2)

        self.output_prob = Sequential()
        self.output_prob.add(Conv2DTranspose(16, 3, activation='relu', padding='same'))
        self.output_prob.add(Conv2DTranspose(16, 3, activation='relu', padding='same'))
        self.output_prob.add(Conv2D(3, 1, activation='softmax', padding='same'))

    def conv_block(self, filters, layers=2):

        net_block = Sequential()

        net_block.add(Conv2D(filters, 3, activation='relu', padding='same'))
        net_block.add(Conv2D(filters, 3, activation='relu', padding='same'))

        if layers == 3:
            net_block.add(Conv2D(filters, 3, activation='relu', padding='same'))

        return net_block

    def deconv_block(self, filters, layers=2):

        net_block = Sequential()

        net_block.add(Conv2DTranspose(filters, 3, activation='relu', padding='same'))

        if layers == 2:
            net_block.add(Conv2DTranspose(filters / 2, 3, activation='relu', padding='same'))

        if layers == 3:
            net_block.add(Conv2DTranspose(filters, 3, activation='relu', padding='same'))
            net_block.add(Conv2DTranspose(filters / 2, 3, activation='relu', padding='same'))

        return net_block

    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.

        :param inputs: images, shape of (num_inputs, 256, 256, 1); during training, the shape is (batch_size, 256, 256, 1)
        :return: probs - a matrix of shape (num_inputs, 256, 256, 3); during training, it would be (batch_size, 256, 256, 3)
        """

        cv1 = self.conv1(inputs)
        pool1, ind1 = maxpool2d_with_argmax(cv1)

        cv2 = self.conv2(pool1)
        pool2, ind2 = maxpool2d_with_argmax(cv2)

        cv3 = self.conv3(pool2)
        pool3, ind3 = maxpool2d_with_argmax(cv3)

        fc = self.connect(pool3)

        unpool1 = maxunpool2d(fc, ind3)
        dcv1 = self.deconv1(unpool1)

        unpool2 = maxunpool2d(dcv1, ind2)
        dcv2 = self.deconv2(unpool2)

        unpool3 = maxunpool2d(dcv2, ind1)
        out = self.output_prob(unpool3)

        return out

    def loss(self, probs, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.

        :param probs: a matrix of shape (batch_size, 256, 256, 3)
        :param labels: during training, matrix of shape (batch_size, 256, 256) containing the train labels
        :return: the loss of the model as a Tensor
        """

        labels = tf.cast(labels, tf.int32)

        class_weights = tf.constant([1, 3, 3])
        class_weights = class_weights / tf.reduce_sum(class_weights)
        sample_weights = tf.gather(class_weights, indices=tf.cast(labels, tf.int32))

        loss = SparseCategoricalCrossentropy()(labels, probs, sample_weights)

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
            show_seg(model, test_inputs[:10], test_labels[:10], 'deconvnet_both' + str(i + 1))


if __name__ == '__main__':
    main()

