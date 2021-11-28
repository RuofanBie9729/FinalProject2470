from scipy.ndimage import rotate
from random import randint

import numpy as np
import tensorflow as tf
import cv2

def aug_rotation(inputs, labels):

    rotation_inputs = np.zeros(inputs.shape)
    rotation_labels = np.zeros(labels.shape)

    for i in range(inputs.shape[2]):

        angle = randint(0, 360)

        rotation_input = rotate(inputs[:, :, i], angle)
        rotation_label = rotate(labels[:, :, i], angle)

        rotation_inputs[:, :, i] = cv2.resize(rotation_input, (256, 256))
        rotation_labels[:, :, i] = cv2.resize(rotation_label, (256, 256))

        return rotation_inputs, rotation_labels

def aug_flip(inputs, labels):

    flip_inputs = np.zeros(inputs.shape)
    flip_labels = np.zeros(labels.shape)

    for i in range(inputs.shape[2]):

        axis = randint(0, 1)

        flip_inputs[:, :, i] = np.flip(inputs[:, :, i], axis)
        flip_labels[:, :, i] = np.flip(labels[:, :, i], axis)

    return flip_inputs, flip_labels

def get_data(input_file_path, output_file_path, aug=None):
    """
    Given two file paths, returns an array of normalized inputs (images) and
    an array of labels (images).
    
    :param input_file_path: file path for the input images, something like 'data/train_img.npy'
    :param output_file_path: file path for the segmentation labels,
    something like 'data/train_lab.npy'
    :param aug: data augmentation method, 'rotate', 'flip' or 'both'.
    :return: normalized tensor of inputs and tensor of labels, where
    inputs are of type np.float64 and has size (num_inputs, 256, 256) and labels
    has size (num_inputs, 256, 256)
    """

    inputs = np.load(input_file_path)
    labels = np.load(output_file_path)

    ## data augmentation
    if aug == "rotate":
        rotation_inputs, rotation_labels = aug_rotation(inputs, labels)
        inputs = np.concatenate([inputs, rotation_inputs], 2)
        labels = np.concatenate([labels, rotation_labels], 2)

    if aug == "flip":
        flip_inputs, flip_labels = aug_flip(inputs, labels)
        inputs = np.concatenate([inputs, flip_inputs], 2)
        labels = np.concatenate([labels, flip_labels], 2)

    if aug == "both":
        rotation_inputs, rotation_labels = aug_rotation(inputs, labels)
        flip_inputs, flip_labels = aug_flip(inputs, labels)

        inputs = np.concatenate([inputs, rotation_inputs, flip_inputs], 2)
        labels = np.concatenate([labels, rotation_labels, flip_labels], 2)

    ## reshape and normalisation
    inputs = tf.transpose(inputs, perm=[2, 0, 1])
    inputs = inputs / 2855

    labels = tf.transpose(labels, perm=[2, 0, 1])

    ## shuffle dataset
    indices = list(range(len(inputs)))
    indices = tf.random.shuffle(indices)

    inputs = tf.gather(inputs, indices)
    labels = tf.gather(labels, indices)

    return inputs, labels