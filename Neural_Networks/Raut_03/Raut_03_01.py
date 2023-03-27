# Raut, Shiska
# 1001_526_329
# 2023_04_02
# Assignment_03_01\

import tensorflow as tf
from tensorflow.keras import layers, models, activations, regularizers, utils
import numpy as np
import matplotlib.pyplot as plt


# creates and saves a annoated heatmap given a matrix and labels
def create_annotated_heatmap(mtrx, x_label, y_label, fig_name):

    # get number of classes
    n_classes = mtrx.shape[0]

    # plt heatmap
    fig, ax = plt.subplots()
    hmap = ax.imshow(mtrx)

    # create annotations
    for i in range(n_classes):
        for j in range(n_classes):
            txt = ax.text(j, i, mtrx[i, j], ha = 'center', va = 'center', color = 'k')

    ax.set_title('Confusion Matrix')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # show, save and close figure
    plt.show()
    plt.savefig(fig_name)
    plt.close(fig)


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

    # view confusion matrix as a heapmap and save figure
    plt.figure(figsize = (10, 6))

    return conf_mtrx


# CNN model object
class ConvolutionalNeuralNetwork(tf.Module):

    def __init__(self, input_dim, optimizer = 'adam'):
        super().__init__()
        self.input_dim = input_dim
        self.optimizer = optimizer
        self.history = None
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

    @tf.function
    def fit_model(self, X_train, Y_train, batch_size, epochs):

        # compile model using optimizer and loss
        self.model.compile(optimizer = self.optimizer, loss = 'categorical_crossentropy')

        # fit model to training data
        self.history = self.model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, validation_split = 0.2)

        # return training history
        return self.history

    @tf.function
    def get_prediction(self, X_test):

        # return model predictions
        return self.model.predict(X_test)

    @tf.function
    def save_model(self, model_name = 'model'):

        # save model with default filename
        self.model.save(model_name, save_format = "h5")


def train_nn_keras(X_train, Y_train, X_test, Y_test, epochs=1, batch_size=4):

    tf.keras.utils.set_random_seed(5368)  # do not remove this line

    # create model
    cnn_model = ConvolutionalNeuralNetwork(X_train.shape[0])

    # train model
    training_history = cnn_model.fit_model(X_train, Y_train, epochs, batch_size)

    # get prediction
    Y_pred = cnn_model.get_prediction(X_test)

    # get confusion matrix
    conf_mtrx = confusion_matrix(Y_test, Y_pred)

    # show and save confusion matrix as a heatmap
    create_annotated_heatmap(conf_mtrx, x_label = 'y_pred', y_label = 'y_true', fig_name = confusion_matrix)
    #
    # Train a convolutional neural network using Keras.
    # X_train: float 32 numpy array [number_of_training_samples, 28, 28, 1]
    # Y_train: float 32 numpy array [number_of_training_samples, 10]  (one hot format)
    # X_test: float 32 numpy array [number_of_test_samples, 28, 28, 1]
    # Y_test: float 32 numpy array [number_of_test_samples, 10]  (one hot format)

    # You will return a list with the following items in the order specified below:
    # - the trained model
    # - the training history (the result of model.fit as an object)
    # - the confusion matrix as numpy array
    # - the output of the model on the test set (the result of model.predict) as numpy array

    return cnn_model, training_history, conf_mtrx, Y_pred