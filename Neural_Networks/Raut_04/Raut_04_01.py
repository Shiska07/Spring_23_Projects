# Raut, Shiska
# 1001_526_329
# 2023_04_16
# Assignment_04_01


# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np
import keras
from keras import Model, Input, layers
# import tensorflow.keras as keras

class CNN(object):
    def __init__(self):
        self.model = None
        self.loss_func = None
        self.metric = None
        optimizer = None
        self.layers = []


    def add_input_layer(self, shape=(2,),name="" ):

        self.layers.append(layers.InputLayer(shape, name))
        

    def append_dense_layer(self, num_nodes,activation="relu",name="",trainable=True):

        self.layers.append(layers.Dense(num_nodes, activation = activation, name = name, trainable = trainable))


    def append_conv2d_layer(self, num_of_filters, kernel_size=3, padding='same', strides=1,
                         activation="Relu",name="",trainable=True):
        self.layers.append(layers.Conv2D(num_of_filters, kernel_size = (kernel_size, kernel_size), padding = padding,
                                                strides = (strides, strides), activation = activation, name = name,
                                                trainable = trainable))

    def append_maxpooling2d_layer(self, pool_size=2, padding="same", strides=2,name=""):

        self.layers.append(layers.MaxPooling2D(pool_size = (pool_size, pool_size), strides = (strides, strides),
                                                    padding = padding, name = name))


    def append_flatten_layer(self,name=""):

        self.layers.append(layers.Flatten())


    def set_training_flag(self,layer_numbers=[],layer_names="",trainable_flag=True):

        # if layer numbers is provided
        if len(layer_numbers) != 0:
            for i in range(len(layer_numbers)):
                self.layers[i + 1].trainable = trainable_flag

        else:
            for i in range(len(layer_names)):
                self.layers[i + 1].trainable = trainable_flag


    def get_weights_without_biases(self,layer_number=None,layer_name=""):

        if layer_number is not None:

            weights = self.layers[layer_number].get_weights()

            # checks if the layer has weights
            if len(weights) >= 1:
                return weights[0]
            else:
                return None
        else:
            for layer in self.layers:
                if layer.name == layer_name:
                    weights = layer.get_weights()

                    # checks if the layer has weights
                    if len(weights) >= 1:
                        return weights[0]
                    else:
                        return None


        """
        This method should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero. Note that layer 0 is the input layer.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0. Note that layer 0 is the input layer
         and it does not have any weights or biases.
         :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
         :return: Weight matrix for the given layer (not including the biases). If the given layer does not have
          weights then None should be returned.
         """


    def get_biases(self,layer_number=None,layer_name=""):

        if layer_number is not None:

            weights = self.layers[layer_number].get_weights()

            # checks if the layer has weights
            if len(weights) == 2:
                return weights[1]
            else:
                return None
        else:
            for layer in self.layers:
                if layer.name == layer_name:
                    weights = layer.get_weights()

                    # checks if the layer has weights
                    if len(weights) == 2:
                        return weights[1]
                    else:
                        return None
        """
        This method should return the biases for layer layer_number.
        layer numbers start from zero. Note that layer 0 is the input layer.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0.Note that layer 0 is the input layer
         and it does not have any weights or biases.
         :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
         :return: biases for the given layer (If the given layer does not have bias then None should be returned)
         """

    def set_weights_without_biases(self,weights,layer_number=None,layer_name=""):
        """
        This method sets the weight matrix for layer layer_number.
        layer numbers start from zero. Note that layer 0 is the input layer.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0. Note that layer 0 is the input layer
         and it does not have any weights or biases.
         :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
         :return: None
         """

    def set_biases(self,biases,layer_number=None,layer_name=""):
        """
        This method sets the biases for layer layer_number.
        layer numbers start from zero. Note that layer 0 is the input layer.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0. Note that layer 0 is the input layer
         and it does not have any weights or biases.
        :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
        :return: none
        """

    def remove_last_layer(self):
        """
        This method removes a layer from the model.
        :return: removed layer
        """

    def load_a_model(self,model_name="",model_file_name=""):
        """
        This method loads a model architecture and weights.
        :param model_name: Name of the model to load. model_name should be one of the following:
        "VGG16", "VGG19"
        :param model_file_name: Name of the file to load the model (if both madel_name and
         model_file_name are specified, model_name takes precedence).
        :return: model
        """

    def save_model(self,model_file_name=""):
        """
        This method saves the current model architecture and weights together in a HDF5 file.
        :param file_name: Name of file to save the model.
        :return: model
        """


    def set_loss_function(self, loss="SparseCategoricalCrossentropy"):
        """
        This method sets the loss function.
        :param loss: loss is a string with the following choices:
        "SparseCategoricalCrossentropy",  "MeanSquaredError", "hinge".
        :return: none
        """

    def set_metric(self,metric):
        """
        This method sets the metric.
        :param metric: metric should be one of the following strings:
        "accuracy", "mse".
        :return: none
        """

    def set_optimizer(self,optimizer="SGD",learning_rate=0.01,momentum=0.0):
        """
        This method sets the optimizer.
        :param optimizer: Should be one of the following:
        "SGD" , "RMSprop" , "Adagrad" ,
        :param learning_rate: Learning rate
        :param momentum: Momentum
        :return: none
        """

    def predict(self, X):
        """
        Given array of inputs, this method calculates the output of the multi-layer network.
        :param X: Input tensor.
        :return: Output tensor.
        """

    def evaluate(self,X,y):
        """
         Given array of inputs and desired ouputs, this method returns the loss value and metrics of the model.
         :param X: Array of input
         :param y: Array of desired (target) outputs
         :return: loss value and metric value
         """
    def train(self, X_train, y_train, batch_size, num_epochs):
        """
         Given a batch of data, and the necessary hyperparameters,
         this method trains the neural network by adjusting the weights and biases of all the layers.
         :param X_train: Array of input
         :param y_train: Array of desired (target) outputs
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :return: list of loss values. Each element of the list should be the value of loss after each epoch.
         """