from __future__ import absolute_import

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
        
        self.batch_size=100
        # Initialize convolutional layers and deconvolutional layer
        self.convnets = tf.keras.Sequential()
        self.deconv = tf.keras.Sequential()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        self.convnets.add(Conv2D(2, 3, activation='relu', padding='same'))
        self.convnets.add(Conv2D(2, 3, activation='relu', padding='same'))
        self.convnets.add(Conv2D(4, 3, activation='relu', padding='same'))
        self.convnets.add(Conv2D(4, 3, activation='relu', padding='same'))
        self.convnets.add(tf.keras.layers.MaxPooling2D())
        self.convnets.add(Conv2D(8, 3, activation='relu', padding='same'))
        self.convnets.add(Conv2D(8, 3, activation='relu', padding='same'))
        self.convnets.add(Conv2D(16, 3, activation='relu', padding='same'))
        self.convnets.add(Conv2D(16, 1, activation='relu', padding='same'))
        self.deconv.add(Conv2DTranspose(3, 40))
        self.deconv.add(Conv2DTranspose(3, 40))
        self.deconv.add(Conv2DTranspose(3, 51))
        

    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        
        :param inputs: images, shape of (num_inputs, 256, 256, 1)
        :return: logits - Flatten predictied image, a matrix of shape (num_inputs, 256*256)
        """
        FCNOutput = self.convnets(inputs)
        logits = self.deconv(FCNOutput)
        prbs = tf.nn.softmax(logits)
        
        return prbs
    
        
        pass

    def loss(self, prbs, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.
        
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        scce = tf.keras.losses.SparseCategoricalCrossentropy()
        
        return tf.math.reduce_mean(scce(labels, prbs))

        pass

    def accuracy(self, prbs, labels):
      return pixel_acc(labels, prbs)
