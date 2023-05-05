import numpy as np
import tensorflow as tf
import keras
from keras import layers, models, activations, regularizers
import matplotlib.pyplot as plt


# creates and saves an annoated heatmap given a matrix and labels
def create_annotated_heatmap(mtrx, x_label, y_label, fig_name):

    # get number of classes
    n_classes = mtrx.shape[0]

    # plt heatmap
    fig, ax = plt.subplots(figsize = (10, 10))
    hmap = ax.matshow(mtrx)

    # create annotations
    for i in range(n_classes):
        for j in range(n_classes):
            txt = ax.text(j, i, mtrx[i, j], ha = 'center', va = 'center', color = 'k')

    ax.set_title('Confusion Matrix')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # save and close figure
    plt.savefig(fig_name)
    plt.close(fig)


# returns a confusion matrix given true and predited values
def confusion_matrix(Y_true, Y_pred, n_classes=10):

    # initialize confusion matrix
    conf_mtrx = np.zeros((n_classes, n_classes), dtype = np.int32)

    for i in range(n_classes):
        for j in range(n_classes):

            conf_mtrx[i, j] = np.sum((Y_true == i) & (Y_pred == j))

    return conf_mtrx

# converts one-hot encoded values into single class value
# given a 2D arr
def reverse_one_hot(arr):

    # conver 2D array to 1D using index of the max value
    labels_arr = np.argmax(arr, axis = 1)

    return labels_arr


# CNN model object
class ConvolutionalNeuralNetwork:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.optimizer = 'adam'
        self.history = None
        self.model = models.Sequential()

        # add layers to the model
        self.model.add(layers.Conv2D(8, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation = 'relu',
                                     kernel_regularizer = regularizers.L2(0.0001), input_shape = self.input_dim))
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

    def fit_model(self, X_train, Y_train, epochs, batch_size):

        # compile model using optimizer and loss
        self.model.compile(optimizer = self.optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])

        # fit model to training data
        self.history = self.model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, validation_split = 0.2)

        # return training history
        return self.history

    def get_prediction(self, X_test):

        # return model predictions
        return self.model.predict(X_test)

    def save_model(self, model_name = 'model.h5'):

        # save model with default filename
        self.model.save(model_name)

    def get_keras_model(self):

        return self.model

def train_nn_keras(X_train, Y_train, X_test, Y_test, epochs=1, batch_size=4):

    tf.keras.utils.set_random_seed(5368)  # do not remove this line

    # create model
    input_size = X_train.shape[1::]
    cnn_model = ConvolutionalNeuralNetwork(input_size)

    # train model and save history
    train_hist = cnn_model.fit_model(X_train, Y_train, epochs, batch_size)

    # save model
    cnn_model.save_model()

    # get prediction
    Y_pred = cnn_model.get_prediction(X_test)

    # reverse one hot encoding 
    Y_pred = reverse_one_hot(Y_pred)
    Y_test = reverse_one_hot(Y_test)

    # get confusion matrix
    conf_mtrx = confusion_matrix(Y_test, Y_pred)

    # show and save confusion matrix as a heatmap
    create_annotated_heatmap(conf_mtrx, x_label = 'Y_pred', y_label = 'Y_true', fig_name = 'confusion_matrix')

    return cnn_model.get_keras_model(), train_hist, conf_mtrx, Y_pred


