# Raut, Shiska
# 1001_526_329
# 2023_03_19
# Assignment_02_01

import numpy as np
import tensorflow as tf

print(tf.__version__)


# returns training and validation arrays according to 'validation_split'
def get_validation_split(X, Y, validation_split):
    # get total no of samples
    n_samp = X.shape[0]

    # set start and stop indices for extracting validation set
    start = int(np.floor(validation_split[0] * n_samp))
    stop = int(np.floor(validation_split[1] * n_samp))

    # extract validation set
    X_val = X[start:stop, :]
    Y_val = Y[start:stop, :]

    # extract training set
    X_train = np.delete(X, np.s_[start:stop], axis=0)
    Y_train = np.delete(Y, np.s_[start:stop], axis=0)

    return X_train.astype('float32'), Y_train.astype('float32'), X_val.astype('float32'), Y_val.astype('float32')


# returns a list of tensors corresponding to weights for each layer
def get_weights_list(input_dim, weights, layers, seed):
    weights_list = []

    # if weights == None, initialize weights for each layer
    if weights is None:
        for n_nodes in layers:
            np.random.seed(seed)
            w_mtx = np.random.randn(input_dim + 1, n_nodes)
            w_mtx = w_mtx.astype('float32')
            # add to weights_list
            weights_list.append(w_mtx)

            # input dim for next layer is the no. of nodes
            # in previous layer + 1 for bias
            input_dim = n_nodes

    # if weights are provided
    else:
        return weights

    return weights_list


# returns svm loss of a batch
def svm_loss(Y, Y_pred):
    # initialize delta value
    n_samples = Y.numpy().shape[0]
    # invert values of Y so delta(1) does not get added to the correct class value
    delta = tf.constant(np.invert((Y.numpy()).astype('bool')), dtype=tf.float32)

    # create a tensor of zeros
    zeros_tnsr = tf.zeros(Y.numpy().shape, dtype=tf.float32)

    Y_st = tf.reshape(tf.reduce_sum(tf.math.multiply(Y_pred, Y), axis=1), shape=(n_samples, 1))

    # get margin values (yj - yst + delta)
    margins = tf.add(tf.subtract(Y_pred, Y_st), delta)

    # get maximum margin
    max_margins = tf.math.maximum(zeros_tnsr, margins)

    # sum over all classes and get total loss
    loss = tf.reduce_mean(tf.reduce_sum(max_margins, axis=1))

    return loss


# returns sum of a batch
def mse_loss(Y, Y_pred):
    # calculate mean squared error
    loss = tf.reduce_mean(tf.reduce_mean(tf.square(tf.subtract(Y, Y_pred)), axis=1), axis=0)

    return loss


# returns cross entropy loss of a batch
def ce_loss(Y, Y_pred):
    # calculate cross entropy loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_pred))

    return loss


# given a list of activation types, returns of list of tf activation functions
def get_activation_func_list(activations):
    act_func_list = []
    for i in range(len(activations)):
        if activations[i] == 'relu':
            act_func_list.append(tf.nn.relu)
        elif activations[i] == 'sigmoid':
            act_func_list.append(tf.math.sigmoid)
        else:
            act_func_list.append(tf.identity)

    return act_func_list


# This function was imported from 'helpers.py' provided by Jason
def generate_batches(X, y, batch_size=32):
    for i in range(0, X.shape[0], batch_size):
        yield X[i:i + batch_size], y[i:i + batch_size]

    # if there's any data left, yield it
    if X.shape[0] % batch_size != 0:
        yield X[-(X.shape[0] % batch_size):], y[-(X.shape[0] % batch_size):]


# creates a single dense layer object
class SingleDenseLayer:

    # layer will be created using weights and activations values
    def __init__(self, weights, layer_activation):
        self.W = tf.Variable(weights[1::, :], dtype=tf.float32, trainable=True)
        self.b = tf.Variable(weights[0, :], dtype=tf.float32, trainable=True)
        self.layer_activation = layer_activation

    def get_layer_output(self, X):
        # calculate net value
        net = tf.matmul(X, self.W) + self.b

        # get activation value
        layer_output = self.layer_activation(net)

        return layer_output

    # update layer weights bad bias given gradient and alpha value
    def update_weights_and_bias(self, gradients_w, gradients_b, alpha):
        # w = w - alpha*gradient (steepest descent)
        self.W.assign_sub(tf.multiply(gradients_w, alpha))
        self.b.assign_sub(tf.multiply(gradients_b, alpha))

        # returns weight

    def get_weights(self):
        return self.W

    # returns bias
    def get_bias(self):
        return self.b


# creates a multi-layer neural network object
class MultiLayerNetwork:

    # network will be constructed using a list of dense layers
    def __init__(self, network_layers):
        self.network_layers = network_layers
        self.n_layers = len(network_layers)

    def get_network_output(self, X):
        for single_layer in self.network_layers:
            # get layer output
            layer_output = single_layer.get_layer_output(X)

            X = layer_output

        # return final output
        return X

    # returns list of weights for the entire model
    def weights(self):

        # initialize list to store weights
        weights_and_bias_list = []
        for single_layer in self.network_layers:
            # concat bias as the first column and append to the list
            weights_and_bias_list.append(single_layer.get_weights())
            weights_and_bias_list.append(single_layer.get_bias())

        return weights_and_bias_list

    # updates weights of the entire network given list of gradients
    # for all layers and alpha value
    def update_weights(self, gradient_list, alpha):

        # get length of the gradient list
        n = len(gradient_list)

        # initialize list to store gradients for individual weights and bias per layer as [w_g, b_g]
        w_and_b_grad_list = []
        for i in range(0, n, 2):
            # gradient for layer weights
            w_g = gradient_list[i]

            # gradient for layer bias
            b_g = gradient_list[i + 1]

            w_and_b_grad_list.append([w_g, b_g])

        # update weights and bias for each layer
        for j, single_layer in enumerate(self.network_layers):
            w_g = w_and_b_grad_list[j][0]
            b_g = w_and_b_grad_list[j][1]
            single_layer.update_weights_and_bias(w_g, b_g, alpha)

        # return a list of updated weights
        return self.weights()

    # returns all the variables(bias included as the first row) of the network as a list of numpy arrays
    def get_weights_list(self):

        # initialize list to store weights
        network_weights_list = []
        for single_layer in self.network_layers:
            # reshape bias as a tensor of rank 2
            b = tf.reshape(single_layer.get_bias(), shape=(1, single_layer.get_bias().numpy().shape[0]))

            # add bias as the first row in weights
            weight_and_bias = tf.concat([b, single_layer.get_weights()], axis=0)

            # concat bias as the first column and append to the list
            network_weights_list.append(weight_and_bias.numpy())

        return network_weights_list


# returns a list of dense layer objects for creating network
def get_network_layers_list(weights_list, act_func_list):
    # initialize list to store layers
    network_layers_list = []

    # create layers using weights and activation functions
    for i, W in enumerate(weights_list):
        network_layers_list.append(SingleDenseLayer(W, act_func_list[i]))

    return network_layers_list


# creates a nn_model object
def create_nn_model(weights_list, act_func_list):
    # create a list of network layers
    network_layers_list = get_network_layers_list(weights_list, act_func_list)

    # create model
    model = MultiLayerNetwork(network_layers_list)

    return model


# returns final model parameters after training a network using batch gradient descent
def train_network(X_train, Y_train, nn_model, alpha, batch_size, loss_func):
    # get no of samples, features and output dim
    n_train_samp, n_feat = X_train.shape
    __, out_dim = Y_train.shape

    # create list to store batches
    training_batches = []

    # generate and save batch in a list
    for X, y in generate_batches(X_train, Y_train, batch_size):
        training_batches.append([X, y])

    # adjust weights after each batch
    for i, batch in enumerate(training_batches):
        # extract batch & convert numpy array to tensor
        X_batch = tf.convert_to_tensor(batch[0], dtype=tf.float32)
        Y_batch = tf.convert_to_tensor(batch[1], dtype=tf.float32)

        # initialize list to store gradient values
        with tf.GradientTape() as g_tape:
            # get predictions
            Y_pred = nn_model.get_network_output(X_batch)

            # get loss value
            loss_val = loss_func(tf.cast(Y_batch, dtype=tf.float32), Y_pred)

        # get gradients   
        gradients_list = g_tape.gradient(loss_val, nn_model.weights())

        # list fo weights and bias tensors occurring alternatively
        updated_weights_and_bias_list = nn_model.update_weights(gradients_list, alpha)

    # list of weights with bias added as the first row
    network_weights_list = nn_model.get_weights_list()

    # return a list of weight tensors 
    return network_weights_list


def multi_layer_nn_tensorflow(X_train, Y_train, layers, activations, alpha, batch_size, epochs=1, loss="svm",
                              validation_split=[0.8, 1.0], weights=None, seed=2):
    # get training and validation data
    X_train, Y_train, X_val, Y_val = get_validation_split(X_train, Y_train, validation_split)

    # initialize weights
    n_train_samp, input_dim = X_train.shape
    n_val_samp, out_dim = Y_val.shape

    # each item in the list as a tf.Variable
    weights_list = get_weights_list(input_dim, weights, layers, seed)

    # initialize list to store error over epochs and final prediction on evaluation data
    err = np.zeros(epochs, dtype=float)

    # create a list of tf_activation modules
    act_func_list = get_activation_func_list(activations)

    # create model
    nn_model = create_nn_model(weights_list, act_func_list)

    # get initial prediction on validation set
    Y_val_pred = nn_model.get_network_output(tf.convert_to_tensor(X_val))

    # get loss function
    if loss == "svm":
        loss_func = svm_loss
    elif loss == "mse":
        loss_func = mse_loss
    else:
        loss_func = ce_loss

    for i in range(epochs):
        weights_list = train_network(X_train, Y_train, nn_model, alpha, batch_size, loss_func)

        # get predictions for evaluation data
        Y_val_pred = nn_model.get_network_output(tf.convert_to_tensor(X_val))

        # get error on prediction data
        err[i] = loss_func(tf.cast(tf.convert_to_tensor(Y_val), dtype=tf.float32), Y_val_pred)

    Y_val_pred = (Y_val_pred.numpy()).astype('float32')

    return [weights_list, err.tolist(), Y_val_pred]
