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
    def __init__(self, name = None):
        self.name = name
        self.model = None
        self.loss = None
        self.metrics = []
        self.optimizer = None
        self.learning_rate = None
        self.momentum = None
        self.layers = []
        self.inputs = None
        self.outputs = None


    def add_input_layer(self, shape=(2,),name="" ):

        self.inputs = layers.InputLayer(shape, name)
        self.layers.append(self.inputs)
        

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


    def get_biases(self,layer_number=None,layer_name=""):

        if layer_number is not None:

            layer_weights = self.layers[layer_number].get_weights()

            # checks if the layer has bias
            if len(layer_weights) == 2:
                return layer_weights[1]
            else:
                return None
        else:
            for layer in self.layers:
                if layer.name == layer_name:
                    layer_weights = layer.get_weights()

                    # checks if the layer has bias
                    if len(layer_weights) == 2:
                        return layer_weights[1]
                    else:
                        return None



    def set_weights_without_biases(self,weights,layer_number=None,layer_name=""):

        # get current weights and replace with new weights
        if layer_number is not None:
            layer_weights = self.layers[layer_number].get_weights()

            # checks if the layer has weights
            if len(layer_weights) >= 1:
                layer_weights[0] = weights
                self.layers[layer_number].set_weights(layer_weights)

        else:
            for layer in self.layers:
                if layer.name == layer_name:
                    layer_weights = layer.get_weights()

                    # checks if the layer has weights
                    if len(layer_weights) >= 1:
                        layer_weights[0] = weights
                        layer.set_weights(layer_weights)


    def set_biases(self,biases,layer_number=None,layer_name=""):

        # get current weights and replace with new bias
        if layer_number is not None:
            layer_weights = self.layers[layer_number].get_weights()

            # checks if the layer has bias
            if len(layer_weights) == 1:
                layer_weights[1] = biases
                self.layers[layer_number].set_weights(layer_weights)

        else:
            for layer in self.layers:
                if layer.name == layer_name:
                    layer_weights = layer.get_weights()

                    # checks if the layer has bias
                    if len(layer_weights) >= 1:
                        layer_weights[0] = biases
                        layer.set_weights(layer_weights)


    def remove_last_layer(self):

        return self.layers.pop()

    def load_a_model(self,model_name="",model_file_name=""):

        # import model
        self.model = keras.models.load_model(model_file_name)

        # set model name
        self.model.name = model_name

        # update model name for CNN object
        self.name = self.model.name


    def save_model(self,model_file_name=""):

        self.model.save(model_file_name)


    def set_loss_function(self, loss="SparseCategoricalCrossentropy"):

        self.loss = loss


    def set_metric(self,metric):

        self.metrics.append(metric)


    def set_optimizer(self,optimizer="SGD",learning_rate=0.01,momentum=0.0):

        if optimizer == "SGD":
            self.optimizer = tf.keras.optimizers.experimental.SGD(learning_rate = learning_rate,
                                                   momentum = momentum)
        elif optimizer == "RMSprop":
            self.optimizer = tf.keras.optimizers.experimental.RMSprop(learning_rate = learning_rate,
                                                                      momentum = momentum)
        elif optimizer == "Adagrad":
            self.optimizer = tf.keras.optimizers.experimental.Adagrad(learning_rate = learning_rate,
                                                                      ema_momentum = momentum,
                                                                      use_ema = True)
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

        # if model has not been initialized
        if self.model is None:

            # input for the firt layer
            x = self.layers[0]

            # iterate through all layers
            for i in range(1, len(self.layers)):

                # output of current layer becomes input for next layer
                x = self.layers[i](x)

            # initialize model
            self.model = Model(inputs = self.inputs, outputs = x)

            # campile model
            self.model.compile(optimizer = self.optimizer, loss = self.loss,
                               metrics = self.metrics)


        """
         Given a batch of data, and the necessary hyperparameters,
         this method trains the neural network by adjusting the weights and biases of all the layers.
         :param X_train: Array of input
         :param y_train: Array of desired (target) outputs
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :return: list of loss values. Each element of the list should be the value of loss after each epoch.
         """