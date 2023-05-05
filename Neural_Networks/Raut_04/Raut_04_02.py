import pytest
import tensorflow as tf
import keras
import numpy as np
from Raut_04_01 import CNN
import os


# tests train() function on mnist data using training and validation loss
def test_training():

    batch_size = 8
    epochs = 30

    # mnist will be used for training
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

    # normalize data
    X_train = (X_train / 255.0 - 0.5).astype(np.float32)
    X_test = (X_test / 255.0 - 0.5).astype(np.float32)

    # reshape data
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    number_of_train_samples_to_use = 200

    X_train = X_train[0:number_of_train_samples_to_use, :]
    Y_train = Y_train[0:number_of_train_samples_to_use]

    Y_train = Y_train.reshape(number_of_train_samples_to_use, 1)

    my_cnn = CNN()

    # "SparseCategoricalCrossentropy" is default loss
    my_cnn.set_loss_function()

    # add input layer
    my_cnn.add_input_layer(shape=(28, 28, 1), name="input")

    # this architecture should lead to an overall decrease in training and validation loss
    my_cnn.append_conv2d_layer(num_of_filters=8, kernel_size=3, strides = 1, padding="same", activation='relu', name="conv1")
    my_cnn.append_conv2d_layer(num_of_filters=16, kernel_size=3, strides=1, padding="same", activation='relu', name="conv2")
    my_cnn.append_maxpooling2d_layer(pool_size=2, strides=2, name="pool1")
    my_cnn.append_conv2d_layer(num_of_filters=32, kernel_size=3, strides=1, padding="same", activation='relu', name="conv3")
    my_cnn.append_conv2d_layer(num_of_filters=64, kernel_size=3, strides=1, padding="same", activation='relu', name="conv4")
    my_cnn.append_maxpooling2d_layer(pool_size=2, strides=2, name="pool2")
    my_cnn.append_flatten_layer(name="flat1")
    my_cnn.append_dense_layer(num_nodes=512, activation="relu", name="dense1")
    my_cnn.append_dense_layer(num_nodes=10, activation="softmax", name="dense2")

    history = my_cnn.train(X_train, Y_train, batch_size, epochs)

    # check if overall training loss decreases
    assert history['loss'][0] > history['loss'][14] > history['loss'][-1]

    # check if overall validation loss decreases
    assert history['val_loss'][0] > history['val_loss'][-1]


# tests evaluate() function on mnist data using training, validation and test accuracy
def test_evaluate():

    batch_size = 8
    epochs = 30

    # mnist will be used for training
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

    # normalize data
    X_train = (X_train / 255.0 - 0.5).astype(np.float32)
    X_test = (X_test / 255.0 - 0.5).astype(np.float32)

    # reshape data
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    number_of_train_samples_to_use = 200
    number_of_test_samples_to_use = 100

    X_train = X_train[0:number_of_train_samples_to_use, :]
    Y_train = Y_train[0:number_of_train_samples_to_use]

    X_test = X_test[0:number_of_test_samples_to_use, :]
    Y_test = Y_test[0:number_of_test_samples_to_use]

    Y_train = Y_train.reshape(number_of_train_samples_to_use, 1)
    Y_test = Y_test.reshape(number_of_test_samples_to_use, 1)

    my_cnn = CNN()

    # "SparseCategoricalCrossentropy" is default loss
    my_cnn.set_loss_function()

    # set metric
    my_cnn.set_metric('accuracy')

    # add input layer
    my_cnn.add_input_layer(shape=(28, 28, 1), name="input")

    # this architecture should lead to an overall decrease in training and validation loss
    my_cnn.append_conv2d_layer(num_of_filters=8, kernel_size=3, strides = 1, padding="same", activation='relu', name="conv1")
    my_cnn.append_conv2d_layer(num_of_filters=16, kernel_size=3, strides=1, padding="same", activation='relu', name="conv2")
    my_cnn.append_maxpooling2d_layer(pool_size=2, strides=2, name="pool1")
    my_cnn.append_conv2d_layer(num_of_filters=32, kernel_size=3, strides=1, padding="same", activation='relu', name="conv3")
    my_cnn.append_conv2d_layer(num_of_filters=64, kernel_size=3, strides=1, padding="same", activation='relu', name="conv4")
    my_cnn.append_maxpooling2d_layer(pool_size=2, strides=2, name="pool2")
    my_cnn.append_flatten_layer(name="flat1")
    my_cnn.append_dense_layer(num_nodes=512, activation="relu", name="dense1")
    my_cnn.append_dense_layer(num_nodes=10, activation="softmax", name="dense2")

    # train model
    history = my_cnn.train(X_train, Y_train, batch_size, epochs)

    # check if overall training accuracy increases
    assert history['accuracy'][0] < history['accuracy'][14] < history['accuracy'][-1]

    # check if overall validation accuracy increases
    assert history['val_accuracy'][0] < history['val_accuracy'][-1]

    # evaluate model
    evaluation_metrics = my_cnn.evaluate(X_test, Y_test)

    # accuracy on test data should be greater than 70%
    assert evaluation_metrics[1] > 0.70
