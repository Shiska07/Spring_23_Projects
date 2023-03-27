# Raut, Shiska
# 1001_526_329
# 2023_04_02
# Assignment_03_01\

import tensorflow as tf
from tensorflow.keras import layers, models, activations, regularizers
import numpy as np
import matplotlib.pyplot as plt


# returns a confusion matrix given true and predited values
def confusion_matrix(y_true, y_pred, n_classes=10):

    # get number of samples
    n_samples = y_true.shape[0]

    # initialize confusion matrix
    conf_mtrx = np.zeros((n_classes, n_classes), dtype = np.float32)

    for c in range(n_classes):

        # get true labels for the class
        true_vals = y_true[:, c]

        # multiply y_pred by y_true values of the class and take the sum
        # over all rows
        conf_mtrx[c, :] = np.sum(true_vals*y_pred, axis = 0)

    # divide by numper of samples to normalize
    conf_mtrx = conf_mtrx/n_samples

    return conf_mtrx


# CNN model object
class ConvolutionalNeuralNetwork:

    def __init__(self, input_dim, optimizer = 'adam'):
        self.input_dim = input_dim
        self.optimizer = optimizer
        self.model = models.Sequential()

        # add layers to the model
        self.model.add(layers.Conv2D(8, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation = 'relu',
                                     kernel_regularizer = regularizers.L2(0.0001), input_size = self.input_dim))
        self.model.add(layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                                     kernel_regularizer = regularizers.L2(0.0001)))
        self.model.add(layers.MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
        self.model.add(layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                                     kernel_regularizer = regularizers.L2(0.0001)))
        self.model.add(layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                                     kernel_regularizer = regularizers.L2(0.0001)))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # flatten the layer
        self.model.add(layers.Flatten())

        # add dense layers
        self.model.add(layers.Dense(512, activation = 'relu', kernel_regularizer = regularizers.L2(0.0001)))
        self.model.add(layers.Dense(10, activation = 'linear', kernel_regularizer = regularizers.L2(0.0001)))

        # add softmax layer
        self.model.add(layers.Activation(activations.softmax))


    def train_model(self, X_train, Y_train, batch_size, epochs):

        # compile model using optimizer and loss
        self.model.compile(optimizer = self.optimizer, loss = 'categorical_crossentropy')

        # fit model to training data
        history = self.model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, validation_split = 0.2)


def train_nn_keras(X_train, Y_train, X_test, Y_test, epochs=1, batch_size=4):
    #
    # Train a convolutional neural network using Keras.
    # X_train: float 32 numpy array [number_of_training_samples, 28, 28, 1]
    # Y_train: float 32 numpy array [number_of_training_samples, 10]  (one hot format)
    # X_test: float 32 numpy array [number_of_test_samples, 28, 28, 1]
    # Y_test: float 32 numpy array [number_of_test_samples, 10]  (one hot format)
    # Assume that the data has been preprocessed (normalized). You do not need to normalize the data.
    # The neural network should have this exact architecture (it's okay to hardcode)
    #
    #
    # The neural network should be trained using the Adam optimizer with default parameters
    # The loss function should be categorical cross-entropy.
    # The number of epochs should be given by the 'epochs' parameter.
    # The batch size should be given by the 'batch_size' parameter.
    # All layers that have weights should have L2 regularization with a regularization strength of 0.0001 (only use kernel regularizer)
    # All other parameters should use keras defaults
    #
    # You should compute the confusion matrix on the test set and return it as a numpy array.
    # You should plot the confusion matrix using the matplotlib function matshow (as heat map) and save it to 'confusion_matrix.png'
    # You should save the keras model to a file called 'model.h5' (do not submit this file). When we test run your program we will check "model.h5"
    # Your program should run uninterrupted and require no user input (including closing figures, etc).
    #
    # You will return a list with the following items in the order specified below:
    # - the trained model
    # - the training history (the result of model.fit as an object)
    # - the confusion matrix as numpy array
    # - the output of the model on the test set (the result of model.predict) as numpy array

    tf.keras.utils.set_random_seed(5368) # do not remove this line
    pass