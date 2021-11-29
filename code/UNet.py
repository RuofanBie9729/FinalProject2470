from preprocessing import get_data
from metrics import IoU, pixel_acc, mean_pixel_acc, show_seg
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

import tensorflow as tf
import numpy as np
import math


class Model(tf.keras.Model):
    def __init__(self):

        super(Model, self).__init__()

        self.batch_size = 32
        self.optimizer = Adam(1e-5)
        self.loss_list = []

        self.encoder1 = self.block(16)
        self.encoder2 = self.block(32, max_pool=True)

        self.decoder1 = self.block(64, max_pool=True, up_sample=True)
        self.decoder2 = self.block(32, up_sample=True)

        # self.encoder1 = self.block(2)
        # self.encoder2 = self.block(4, max_pool=True)
        # self.encoder3 = self.block(8, max_pool=True)

        # self.decoder1 = self.block(16, max_pool=True, up_sample=True)
        # self.decoder2 = self.block(8, up_sample=True)
        # self.decoder3 = self.block(4, up_sample=True)

        self.output_prob = self.block(16, output=True)

    def block(self, filters, max_pool=False, up_sample=False, output=False):

        net_block = Sequential()

        if max_pool:
            net_block.add(MaxPool2D())

        net_block.add(Conv2D(filters, 3, activation='relu', padding='same'))
        net_block.add(Conv2D(filters, 3, activation='relu', padding='same'))

        if up_sample:
            net_block.add(UpSampling2D())
            net_block.add(Conv2D(filters / 2, 2, activation='relu', padding='same'))

        if output:
            net_block.add(Conv2D(2, 1, activation='softmax', padding='same'))

        return net_block

    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.

        :param inputs: images, shape of (num_inputs, 256, 256, 1); during training, the shape is (batch_size, 256, 256, 1)
        :return: probs - a matrix of shape (num_inputs, 256, 256, 3); during training, it would be (batch_size, 256, 256, 3)
        """

        encode1 = self.encoder1(inputs)
        encode2 = self.encoder2(encode1)

        decode1 = self.decoder1(encode2)
        merge1 = concatenate([encode2, decode1], 3)

        decode2 = self.decoder2(merge1)
        merge2 = concatenate([encode1, decode2], 3)

        out = self.output_prob(merge2)

        # encode1 = self.encoder1(inputs)
        # encode2 = self.encoder2(encode1)
        # encode3 = self.encoder3(encode2)

        # decode1 = self.decoder1(encode3)
        # merge1 = concatenate([encode3, decode1], 3)

        # decode2 = self.decoder2(merge1)
        # merge2 = concatenate([encode2, decode2], 3)

        # decode3 = self.decoder3(merge2)
        # merge3 = concatenate([encode1, decode3], 3)

        # out = self.output_prob(merge3)

        return out

    def loss(self, probs, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.

        :param probs: a matrix of shape (batch_size, 256, 256, 3)
        :param labels: during training, matrix of shape (batch_size, 256, 256) containing the train labels
        :return: the loss of the model as a Tensor
        """

        labels = tf.cast(labels, tf.int32)

        # class_weights = tf.constant([np.sum(labels==0), np.sum(labels==1), np.sum(labels==2)])
        # class_weights = 1-class_weights/tf.reduce_sum(class_weights)
        class_weights = tf.constant([1, 2])
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
        iou += IoU(batch_labels, probs).numpy()
        acc += pixel_acc(batch_labels, probs)
        mean_acc += mean_pixel_acc(batch_labels, probs)

    return iou / n_batch, acc / n_batch, mean_acc / n_batch


def main():
    train_inputs, train_labels = get_data('/gdrive/My Drive/CSCI2470 FinalProject Dataset/train_img_r.npy',
                                          '/gdrive/My Drive/CSCI2470 FinalProject Dataset/train_lab_r.npy', 'both')
    train_inputs = tf.reshape(train_inputs, (train_inputs.shape[0], 256, 256, 1))
    train_labels = tf.where(tf.cast(train_labels, tf.int32) == 0, 0, 1)
    test_inputs, test_labels = get_data('/gdrive/My Drive/CSCI2470 FinalProject Dataset/test_img_r.npy',
                                        '/gdrive/My Drive/CSCI2470 FinalProject Dataset/test_lab_r.npy')
    test_inputs = tf.reshape(test_inputs, (test_inputs.shape[0], 256, 256, 1))
    test_labels = tf.where(tf.cast(test_labels, tf.int32) == 0, 0, 1)

    # create model
    model = Model()

    # train
    for i in range(50):
        train(model, train_inputs, train_labels)
        print(f"Train Epoch: {i} \tLoss: {np.mean(model.loss_list):.6f}")
        iou, acc, mean_acc = test(model, test_inputs, test_labels)
        print(f"--IoU: {iou:.6f}  --pixel accuracy: {acc:.6f}  --mean pixel accuracy: {mean_acc:.6f}")
        if (i + 1) % 10 == 0:
            show_seg(model, test_inputs[:10], test_labels[:10], 'unet_no_aug' + str(i + 1))


if __name__ == '__main__':
    main()
