from __future__ import absolute_import
from metrics import IoU, pixel_acc, mean_pixel_acc, show_seg
from preprocessing import get_data
from tensorflow.keras.metrics import MeanIoU

import tensorflow as tf
import tensorflow.keras as keras
from keras import backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose, Dropout
from tensorflow.math import exp, sqrt, square
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np
import random
import math

def get_layer_outputs(model, layer_name, input_data, learning_phase=1):
    outputs   = [layer.output for layer in model.layers if layer_name in layer.name]
    layers_fn = K.function([model.input, K.learning_phase()], outputs)
    return layers_fn([input_data, learning_phase])




class FCN(tf.keras.Model):
    def __init__(self, fcn_32s = False, fcn_16s = False):
        """
        This model class will contain the architecture for our FCN that
        implements image segmentation.
        """
        super(FCN, self).__init__()

        self.batch_size = 1
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.fcn_32s = fcn_32s
        self.fcn_16s = fcn_16s
        # Initialize convolutional layers and deconvolutional layer
        self.softmax = Conv2D(3, 1, activation='softmax', padding='same')
		
    def vgg16_base(self):
        #input_layer = tf.keras.Input(shape=(256, 256, 3), name="input")
        vgg16 = tf.keras.Sequential()

        vgg16.add(Conv2D(64, 3, activation='relu', padding='same', input_shape=(256, 256, 3), name='conv11'))
        vgg16.add(Conv2D(64, 3, activation='relu', padding='same', name='conv12'))
        vgg16.add(tf.keras.layers.MaxPooling2D())
        vgg16.add(Conv2D(128, 3, activation='relu', padding='same', name='conv21'))
        vgg16.add(Conv2D(128, 3, activation='relu', padding='same', name='conv22'))
        vgg16.add(tf.keras.layers.MaxPooling2D())
        vgg16.add(Conv2D(256, 3, activation='relu', padding='same', name='conv31'))
        vgg16.add(Conv2D(256, 3, activation='relu', padding='same', name='conv32'))
        vgg16.add(Conv2D(256, 3, activation='relu', padding='same', name='conv33'))
        vgg16.add(tf.keras.layers.MaxPooling2D(name='pool3'))
    
        vgg16.add(Conv2D(512, 3, activation='relu', padding='same', name='conv41'))
        vgg16.add(Conv2D(512, 3, activation='relu', padding='same', name='conv42'))
        vgg16.add(Conv2D(512, 3, activation='relu', padding='same', name='conv43'))
        vgg16.add(tf.keras.layers.MaxPooling2D(name='pool4'))
       
        vgg16.add(Conv2D(512, 3, activation='relu', padding='same', name='conv51'))
        vgg16.add(Conv2D(512, 3, activation='relu', padding='same', name='conv52'))
        vgg16.add(Conv2D(512, 3, activation='relu', padding='same', name='conv53'))
        vgg16.add(tf.keras.layers.MaxPooling2D())
        vgg16.add(Conv2D(4096, 7, activation='relu', padding='same', name='dense1'))
        vgg16.add(Dropout(0.2))
        vgg16.add(Conv2D(4096, 1, activation='relu', padding='same', name='dense2')) 
        vgg16.add(Dropout(0.2, name='conv7'))

        vgg16.add(Conv2D(1000, 1, activation='relu', padding='same', name='pred'))
        #output = vgg16(input_layer)
        return tf.keras.Model(vgg16.input, vgg16.output)
   
    def Model(self, vgg16):   
        vgg16Output = vgg16.get_layer('conv7').output
        vgg16pool4 = vgg16.get_layer('pool4').output
        vgg16pool3 = vgg16.get_layer('pool3').output
        if self.fcn_32s:
            predict1 = Conv2D(3, 1, activation='relu', padding='same')(vgg16Output)
            FCNoutput = Conv2DTranspose(3, 64, 32, padding='same')(predict1)		
        elif self.fcn_16s:
            predict1 = Conv2D(3, 1, activation='relu', padding='same')(vgg16Output)
            predict1 = Conv2DTranspose(3, 4, 2, padding='same')(predict1)
            convout2 = Conv2D(3, 1, activation='relu', padding='same')(vgg16pool4)
            FCNoutput = tf.add(predict1, convout2)
            FCNoutput = Conv2DTranspose(3, 32, 16, padding='same')(FCNoutput)
        else:
            predict1 = Conv2D(3, 1, activation='relu', padding='same')(vgg16Output)
            predict1 = Conv2DTranspose(3, 4, 2, padding='same')(predict1)
            convout2 = Conv2D(3, 1, activation='relu', padding='same')(vgg16pool4)
            FCNoutput = tf.add(predict1, convout2)
            FCNoutput = Conv2DTranspose(3, 4, 2, padding='same')(FCNoutput)
            convout1 = Conv2D(3, 1, activation='relu', padding='same')(vgg16pool3)
            FCNoutput = tf.add(FCNoutput, convout1)
            FCNoutput = Conv2DTranspose(3, 16, 8, padding='same')(FCNoutput)
        FCNoutput = self.softmax(FCNoutput)
        return tf.keras.Model(vgg16.input, FCNoutput)	

    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 256, 256, 1)
        :return: logits - Flatten predictied image, a matrix of shape (num_inputs, 256*256)
        """
        base_model = self.vgg16_base()
        FCNmodel = self.Model(base_model)
        FCNmodel.get_layer('conv11').trainable=False
        FCNmodel.get_layer('conv12').trainable=False
        FCNmodel.get_layer('conv21').trainable=False
        FCNmodel.get_layer('conv22').trainable=False
        FCNmodel.get_layer('conv31').trainable=False
        FCNmodel.get_layer('conv32').trainable=False
        FCNmodel.get_layer('conv33').trainable=False
        FCNmodel.get_layer('conv41').trainable=False
        FCNmodel.get_layer('conv42').trainable=False
        FCNmodel.get_layer('conv43').trainable=False
        FCNmodel.get_layer('conv51').trainable=False
        FCNmodel.get_layer('conv52').trainable=False
        FCNmodel.get_layer('conv53').trainable=False
        FCNmodel.get_layer('dense1').trainable=False
        FCNmodel.get_layer('dense2').trainable=False
        prbs = FCNmodel(inputs)

        return prbs
 
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
        #print(tf.reduce_mean(loss))
        return tf.reduce_mean(loss)


class BilinearInitializer(tf.keras.initializers.Initializer):
    '''Initializer for Conv2DTranspose to perform bilinear interpolation on each channel.'''
    def __call__(self, shape, dtype=None, **kwargs):
        kernel_size, _, filters, _ = shape
        arr = np.zeros((kernel_size, kernel_size, filters, filters))
        ## make filter that performs bilinear interpolation through Conv2DTranspose
        upscale_factor = (kernel_size+1)//2
        if kernel_size % 2 == 1:
            center = upscale_factor - 1
        else:
            center = upscale_factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        kernel = (1-np.abs(og[0]-center)/upscale_factor) * \
                 (1-np.abs(og[1]-center)/upscale_factor) # kernel shape is (kernel_size, kernel_size)
        for i in range(filters):
            arr[..., i, i] = kernel
        return tf.convert_to_tensor(arr, dtype=dtype)


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
            model.loss_list.append(loss)

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


def main(arg = None):

    inputs, train_labels = get_data('train_img_r.npy', 'train_lab_r.npy', arg)
    train_inputs = np.zeros((inputs.shape[0], 256, 256, 3))
    train_inputs[:, :, :, 0] = inputs
    train_inputs[:, :, :, 1] = inputs
    train_inputs[:, :, :, 2] = inputs
    train_inputs = tf.convert_to_tensor(train_inputs)
    inputs, test_labels = get_data('test_img_r.npy', 'test_lab_r.npy')
    test_inputs = np.zeros((inputs.shape[0], 256, 256, 3))
    test_inputs[:, :, :, 0] = inputs
    test_inputs[:, :, :, 1] = inputs
    test_inputs[:, :, :, 2] = inputs
    test_inputs = tf.convert_to_tensor(test_inputs)

    # create model


    model = FCN()
    #base_model = self.vgg16_base()
    vgg16 = tf.keras.applications.vgg16.VGG16(weights='imagenet')
    weight_list = vgg16.get_weights()
    weight_list[26] = weight_list[26].reshape(7, 7, 512, 4096)
    weight_list[28] = weight_list[28].reshape(1, 1, 4096, 4096)
    weight_list[30] = weight_list[30].reshape(1, 1, 4096, 1000)
    model.vgg16_base().set_weights(weight_list)
    del weight_list	


    # train
    for i in range(50):
        train(model, train_inputs, train_labels)
        print(f'Train Epoch: {i}  Loss: {np.mean(model.loss_list):.6f}')
        iou, acc, mean_acc = test(model, test_inputs, test_labels)
        print(f'--IoU: {iou:.6f}  --pixel accuracy: {acc:.6f}  --mean pixel accuracy: {mean_acc:.6f}')

        if (i + 1) % 10 == 0:
            show_seg(model, test_inputs[:10], test_labels[:10], 'fcn_add' + arg + str(i + 1))

if __name__ == '__main__':
    main('None')
    main('both')
    main('flip')
    main('rotate')
