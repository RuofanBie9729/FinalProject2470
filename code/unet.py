from preprocessing import get_data
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

import tensorflow as tf
import numpy as np
import math


class Model(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that
        classifies images. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()

        self.batch_size = 64
        self.optimizer = Adam(learning_rate=1e-5)

    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.

        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """

        conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = MaxPool2D()(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = MaxPool2D()(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
        pool3 = MaxPool2D()(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
        pool4 = MaxPool2D()(conv4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
        up5 = UpSampling2D()(conv5)
        up5 = Conv2D(512, 2, activation='relu', padding='same')(up5)

        merge6 = concatenate([conv4, up5], 3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)
        up6 = UpSampling2D()(conv6)
        up6 = Conv2D(256, 2, activation='relu', padding='same')(up6)

        merge7 = concatenate([conv3, up6], 3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)
        up7 = UpSampling2D()(conv7)
        up7 = Conv2D(128, 2, activation='relu', padding='same')(up7)

        merge8 = concatenate([conv2, up7], 3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)
        up8 = UpSampling2D()(conv8)
        up8 = Conv2D(64, 2, activation='relu', padding='same')(up8)

        merge9 = concatenate([conv1, up8], 3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

        conv10 = Conv2D(3, 1, activation='relu', padding='same')(conv9)

        return conv10

    def loss(self, preds, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.

        :param preds: a matrix of shape (batch_size, width, height, 3)
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, width, height) containing the train labels
        :return: the loss of the model as a Tensor
        """
        loss = SparseCategoricalCrossentropy()(labels, preds)

        return tf.reduce_mean(loss)

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.

        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT

        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


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
            preds = model(batch_inputs)
            loss = model.loss(preds, batch_labels)
            loss_list.append(loss.numpy())

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss_list


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly
    flip images or do any extra preprocessing.

    :param test_inputs: test data (all images to be tested),
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """
    preds = model(test_inputs)
    acc = model.accuracy(preds, test_labels)

    return acc.numpy()

def main():
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs.

    CS1470 students should receive a final accuracy
    on the testing examples for cat and dog of >=70%.

    CS2470 students should receive a final accuracy
    on the testing examples for cat and dog of >=75%.

    :return: None
    '''
    train_inputs, train_labels = get_data('data/train_img.npy', 'data/train_lab.npy', aug='both')
    train_inputs = tf.reshape(train_inputs, (train_inputs.shape[0], train_inputs.shape[1], train_inputs.shape[2], 1))
    # test_inputs, test_labels = get_data('data/test_img.npy', 'data/test_lab.npy')

    # create model
    model = Model()

    # train
    for i in range(5):
        loss_list = train(model, train_inputs, train_labels)
        print(np.mean(loss_list))


if __name__ == '__main__':
    main()
