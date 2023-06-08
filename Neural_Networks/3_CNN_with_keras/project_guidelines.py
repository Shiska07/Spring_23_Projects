import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def confusion_matrix(y_true, y_pred, n_classes=10):
    #
    # Compute the confusion matrix for a set of predictions
    #
    # the shape of the confusion matrix should be (n_classes, n_classes)
    # The (i, j)th entry should be the number of times an example with true label i was predicted label j
    #
    # Do not use any libraries to use this function (e.g. sklearn.metrics.confusion_matrix, or tensorflow.math.confusion_matrix)
    pass


def train_nn_keras(X_train, Y_train, X_test, Y_test, epochs=1, batch_size=4):
    #
    # Train a convolutional neural network using Keras.
    # X_train: float 32 numpy array [number_of_training_samples, 28, 28, 1]
    # Y_train: float 32 numpy array [number_of_training_samples, 10]  (one hot format)
    # X_test: float 32 numpy array [number_of_test_samples, 28, 28, 1]
    # Y_test: float 32 numpy array [number_of_test_samples, 10]  (one hot format)
    # Assume that the data has been preprocessed (normalized). You do not need to normalize the data.
    # The neural network should have this exact architecture (it's okay to hardcode)
    # - Convolutional layer with 8 filters, kernel size 3 by 3, stride 1 by 1, padding 'same', and ReLU activation
    # - Convolutional layer with 16 filters, kernel size 3 by 3, stride 1 by 1, padding 'same', and ReLU activation
    # - Max pooling layer with pool size 2 by 2 and stride 2 by 2
    # - Convolutional layer with 32 filters, kernel size 3 by 3, stride 1 by 1, padding 'same', and ReLU activation
    # - Convolutional layer with 64 filters, kernel size 3 by 3 , stride 1 by 1, padding 'same', and ReLU activation
    # - Max pooling layer with pool size 2 by 2 and stride 2 by 2
    # - Flatten layer
    # - Dense layer with 512 units and ReLU activation
    # - Dense layer with 10 units with linear activation
    # - a softmax layer
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
    pass