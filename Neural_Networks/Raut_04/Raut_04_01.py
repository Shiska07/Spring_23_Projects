# Raut, Shiska
# 1001_526_329
# 2023_04_16
# Assignment_04_01


# %tensorflow_version 2.x
import os
import tensorflow as tf
import numpy as np
import keras
from keras import Model, Input, layers
# import tensorflow.keras as keras

class CNN(object):
    def __init__(self, name = None):
        self.name = name                        # model name
        self.cnn_model = None                   # keras cnn_model object
        self.cnn_layers = []  # list of layers in the network
        self.input_layer = False  # tracks whether input layer is present
        self.loss = None                        # loss type
        self.metrics = []                       # list of evaluation metrics
        self.optimizer = None                   # optimizer type
        self.learning_rate = None               # optimizer learning rate
        self.momentum = None                    # optimizer momentum
        self.model_history = None               # model history
        self.model_compiled = False             # tracks if the model has been compiled


    def add_input_layer(self, shape=(2,),name="" ):

        # if input layer is present
        if self.input_layer:
            # remove input later
            self.cnn_layers.pop(0)
            self.cnn_layers.insert(0, keras.Input(shape = shape, name = name))
        else:
            self.cnn_layers.append(keras.Input(shape = shape, name = name))
            self.input_layer = True


    def append_dense_layer(self, num_nodes,activation="relu",name="",trainable=True):

        self.cnn_layers.append(layers.Dense(num_nodes, activation = activation,
                                            name = name, trainable = trainable))

        # compile model after adding a layer to initialize weights
        self.compile_model()
        self.model_compiled = True


    def append_conv2d_layer(self, num_of_filters, kernel_size=3, padding='same', strides=1,
                         activation="Relu",name="",trainable=True):

        conv_layer = layers.Conv2D(num_of_filters, kernel_size = kernel_size, padding = padding,
                                                strides = strides, activation = activation, name = name,
                                                trainable = trainable)
        self.cnn_layers.append(conv_layer)

        # compile model after adding a layer to initialize weights
        self.compile_model()
        self.model_compiled = True

        return conv_layer


    def append_maxpooling2d_layer(self, pool_size=2, padding="same", strides=2,name=""):

        pooling_layer = layers.MaxPooling2D(pool_size = pool_size, strides = strides,
                                                    padding = padding, name = name)
        self.cnn_layers.append(pooling_layer)

        # compile model after adding a layer
        self.compile_model()
        self.model_compiled = True

        return pooling_layer


    def append_flatten_layer(self,name=""):

        flatten_layer = layers.Flatten()
        self.cnn_layers.append(flatten_layer)

        # compile model after adding a layer
        self.compile_model()
        self.model_compiled = True

        return flatten_layer


    def set_training_flag(self,layer_numbers=[],layer_names="",trainable_flag=True):

        if type(layer_numbers) is list:
            if  len(layer_numbers) != 0:
                for i in range(len(layer_numbers)):
                    self.cnn_layers[i + 1].trainable = trainable_flag

            # if the list of layer numbers is empty
            else:
                 # if layer_names is a list
                 if type(layer_names) is list:
                     # kayer_names is a non-empty list
                     if len(layer_names) != 0:
                         for layer_name in layer_names:
                             for cnn_layer in self.cnn_layers:
                                 if cnn_layer.name == layer_name:
                                     cnn_layer.trainable = trainable_flag

                 # if layer_names is a non-empty string
                 elif len(layer_names) != 0:
                     for cnn_layer in self.cnn_layers:
                         if cnn_layer.name == layer_names:
                             cnn_layer.trainable = trainable_flag

        # if layer_numbers is a single number
        elif layer_numbers >= 0:
            self.cnn_layers[layer_numbers].trainable = trainable_flag


        # recompile model as flags have been modified
        self.compile_model()
        self.model_compiled = True


    def get_weights_without_biases(self,layer_number=None,layer_name=""):

        if layer_number is not None:

            # if layer_number = 0 (input_layer), return None
            if layer_number == 0:
                return None

            weights = self.cnn_layers[layer_number].get_weights()

            # checks if the layer has weights
            if len(weights) >= 1:
                return weights[0]
            else:
                return None
        else:
            for cnn_layer in self.cnn_layers:
                if cnn_layer.name == layer_name:
                    weights = cnn_layer.get_weights()

                    # checks if the layer has weights
                    if len(weights) >= 1:
                        return weights[0]
                    else:
                        return None


    def get_biases(self,layer_number=None,layer_name=""):

        if layer_number is not None:

            # if layer_number = 0 (input_layer), return None
            if layer_number == 0:
                return None

            layer_weights = self.cnn_layers[layer_number].get_weights()

            # checks if the layer has bias
            if len(layer_weights) == 2:
                return layer_weights[1]
            else:
                return None
        else:
            for cnn_layer in self.cnn_layers:
                if cnn_layer.name == layer_name:
                    layer_weights = cnn_layer.get_weights()

                    # checks if the layer has bias
                    if len(layer_weights) == 2:
                        return layer_weights[1]
                    else:
                        return None


    def set_weights_without_biases(self,weights,layer_number=None,layer_name=""):

        # get current weights and replace with new weights
        if layer_number is not None:

            # if layer_number = 0 (input_layer), return None
            if layer_number == 0:
                return None

            layer_weights = self.cnn_layers[layer_number].get_weights()

            # checks if the layer has weights
            if len(layer_weights) >= 1:

                # remove weight from the list
                layer_weights.pop(0)

                # insert new weight
                layer_weights.insert(0, weights)

                # set new weight
                self.cnn_layers[layer_number].set_weights(layer_weights)

        else:
            for cnn_layer in self.cnn_layers:
                if cnn_layer.name == layer_name:
                    layer_weights = cnn_layer.get_weights()

                    # checks if the layer has weights
                    if len(layer_weights) >= 1:

                        # remove weight from the list
                        layer_weights.pop(0)

                        # insert new weight
                        layer_weights.insert(0, weights)

                        # set new weight
                        cnn_layer.set_weights(layer_weights)

        # recompile model as weights have been changed
        self.compile_model()
        self.model_compiled = True


    def set_biases(self,biases,layer_number=None,layer_name=""):

        # get current weights and replace with new bias
        if layer_number is not None:

            # if layer_number = 0 (input_layer), return None
            if layer_number == 0:
                return None

            layer_weights = self.cnn_layers[layer_number].get_weights()

            # checks if the layer has bias
            if len(layer_weights) == 2:

                # remove weight from the list
                layer_weights.pop(1)

                # insert new weight
                layer_weights.insert(1, biases)

                # set new weight
                self.cnn_layers[layer_number].set_weights(layer_weights)

        else:
            for cnn_layer in self.cnn_layers:
                if cnn_layer.name == layer_name:
                    layer_weights = cnn_layer.get_weights()

                    # checks if the layer has bias
                    if len(layer_weights) == 2:

                        # remove weight from the list
                        layer_weights.pop(1)

                        # insert new weight
                        layer_weights.insert(1, biases)

                        # set new weight
                        cnn_layer.set_weights(layer_weights)

        # recompile model as weights have been changed
        self.compile_model()
        self.model_compiled = True


    def remove_last_layer(self):

        # remove last layer and recompile model
        removed_layer = self.cnn_layers.pop()
        self.compile_model()
        self.model_compiled = True

        return removed_layer


    def load_a_model(self,model_name="",model_file_name=""):

        # if model_name is provided
        if len(model_name) != 0:

            if model_name == "VGG16":

                # load model from keras
                self.cnn_model = tf.keras.applications.VGG16()

            else:
                # load model from keras
                self.cnn_model = tf.keras.applications.VGG19()

        # if file path is provided
        elif len(model_file_name) != 0:

            self.cnn_model = keras.models.load_model(model_file_name)


        # empty the current list of layers
        if len(self.cnn_layers) != 0:
            self.cnn_layers.clear()
            self.input_layer = False
            self.model_compiled = False


        self.add_input_layer(shape = self.cnn_model.input_shape[1::])

        # add all the model's layers to the list
        for i, layer in enumerate(self.cnn_model.layers):
            # append all layers except the first input layer
            if i != 0:
                self.cnn_layers.append(layer)

        # compile model
        self.compile_model()
        self.model_compiled = True

        # update model name for CNN object
        self.name = self.cnn_model.name

        return self.cnn_model


    def save_model(self,model_file_name=""):

        # recompile model before saving
        self.compile_model()
        self.model_compiled = True

        self.cnn_model.save(model_file_name, save_format = "h5")

        return self.cnn_model


    def set_loss_function(self, loss="SparseCategoricalCrossentropy"):

        self.loss = loss


    def set_metric(self,metric):

        self.metrics.append(metric)


    def set_optimizer(self,optimizer="SGD",learning_rate=0.01,momentum=0.0):

        if optimizer == "SGD":
            self.optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate,
                                                   momentum = momentum)
        elif optimizer == "RMSprop":
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate = learning_rate,
                                                                      momentum = momentum)
        elif optimizer == "Adagrad":
            self.optimizer = tf.keras.optimizers.Adagrad(learning_rate = learning_rate,
                                                                      ema_momentum = momentum,
                                                                      use_ema = True)


    def compile_model(self):

        # input for the firt layer
        x = self.cnn_layers[0]

        # iterate through all layers
        for i in range(1, len(self.cnn_layers)):
            # output of current layer becomes input for next layer
            x = self.cnn_layers[i](x)

        # initialize model
        self.cnn_model = Model(inputs=self.cnn_layers[0], outputs=x)

        # set defalut optimizer and loss functions if unintialized
        if self.optimizer is None:
            self.set_optimizer()
        if self.loss is None:
            self.set_loss_function()

        # campile model
        self.cnn_model.compile(optimizer=self.optimizer, loss=self.loss,
                               metrics=self.metrics)


    def predict(self, X):

        # model has not been compiled
        if not self.model_compiled:
            self.compile_model()
            self.model_compiled = True

        return self.cnn_model.predict(X)


    def evaluate(self,X,y):

        # model has not been compiled
        if not self.model_compiled:
            self.compile_model()
            self.model_compiled = True

        results = self.cnn_model.evaluate(X, y)

        return results


    def train(self, X_train, y_train, batch_size, num_epochs):

        # model has not been compiled
        if not self.model_compiled:
            self.compile_model()
            self.model_compiled = True

        # train model with validation split of 0.2
        self.model_history = self.cnn_model.fit(X_train, y_train, batch_size = batch_size,
                                                epochs = num_epochs, validation_split = 0.2)

        # return model history
        return self.model_history.history


