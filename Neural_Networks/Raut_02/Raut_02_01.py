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
    start = int(np.floor(validation_split[0]*n_samp))
    stop = int(np.floor(validation_split[1]*n_samp))

    # extract validation set
    X_val = X[start:stop, :]
    Y_val = Y[start:stop, :]

    # extract training set
    X_train = np.delete(X, np.s_[start:stop], axis = 0)
    Y_train = np.delete(Y, np.s_[start:stop], axis = 0)

    return X_train.astype('float32'), Y_train.astype('float32'), X_val.astype('float32'), Y_val.astype('float32')


# returns a list of tensors correspong to weights for each layer
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
    # invert values to make sure it is 0 for correct class label
    delta = tf.bitwise.invert(Y)

    # create a tensor of zeros
    zeros_tnsr = tf.zeros((Y.numpy().shape), dtype = tf.float32)

    # get margin values (yj - yst + delta)
    margins = tf.add(tf.subtract(Y_pred, tf.reduce_sum(tf.math.multiply(Y_pred, Y), axis = 1)), delta)

    # get maximum margin
    max_margins = tf.math.maximum(zeros_tnsr, margins)

    # sum over all classes and get total loss
    svm_loss = tf.reduce_mean(tf.reduce_sum(max_margins, axis = 1))

    return svm_loss


# returns sum of a batch
def mse_loss(Y, Y_pred):

    # caluclate mean squared error
    mse_loss = tf.reduce_mean(tf.square(tf.subtract(Y, Y_pred)))

    return mse_loss


# returns cross entropy loss of a batch
def ce_loss(Y, Y_pred):

    # calculate cross entropy loss
    ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = Y_pred))

    return ce_loss


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


# creates a single dense layer object
class SingleDenseLayer(tf.Module):

    # layer will be created using weights and activations values
    def __init__(self, weights, layer_activation):
        self.W = tf.Variable(weights[1::,:], dtype = tf.float32, trainable = True)
        self.b = tf.Variable(weights[0,:], dtype = tf.float32, trainable = True)
        self.layer_activation = layer_activation

    def get_layer_output(self, X):

        # caluclate net value
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
class MultiLayerNetwork(tf.Module):

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
            b_g = gradient_list[i+1]

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
            b = tf.reshape(single_layer.get_bias(), shape = (1, single_layer.get_bias().numpy().shape[0]))

            # add bias as the first row in weights
            weight_and_bias = tf.concat([b, single_layer.get_weights()], axis = 0)

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

    # get number of batches
    n_batches = int(np.ceil(n_train_samp/batch_size))

    for i in range(n_batches):

        # get indices for extracting a single batch
        start = i*batch_size
        end = (i+1)*batch_size

        if end > n_train_samp:
            end = n_train_samp

        # extract batch & convert numpy array to tensor
        X_batch = tf.convert_to_tensor(X_train[start:end,:], dtype = tf.float32)
        Y_batch = tf.convert_to_tensor(Y_train[start:end,:], dtype = tf.float32)
        
        # initialize list to store gradient values
        with tf.GradientTape() as g_tape:
            
            # get predictions
            Y_pred = nn_model.get_network_output(X_batch)

            # get loss value
            loss_val = loss_func(tf.cast(Y_batch, dtype = tf.float32), Y_pred)

        # get gradients   
        gradients_list = g_tape.gradient(loss_val, nn_model.weights())

        # list fo weights and bias tensors occuring alternatively
        updated_weights_and_bias_list = nn_model.update_weights(gradients_list, alpha)

    # list of weights with bias added as the first row
    network_weights_list = nn_model.get_weights_list()

    # return a list of weight tensors 
    return network_weights_list


def multi_layer_nn_tensorflow(X_train, Y_train, layers, activations, alpha, batch_size, epochs=1, loss="svm",
                              validation_split=[0.8,1.0], weights=None, seed=2):

    # get training and validation data
    X_train, Y_train, X_val, Y_val = get_validation_split(X_train, Y_train, validation_split)

    # initialize weights
    n_train_samp, input_dim = X_train.shape
    n_val_samp, out_dim = Y_val.shape

    # each item in the list as a tf.Variable
    weights_list = get_weights_list(input_dim, weights, layers, seed)

    # initialize list to store error over epochs and final prediction on evaluation data
    err = np.zeros((1, epochs), dtype = float)

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
        err[:,i] = loss_func(tf.cast(tf.convert_to_tensor(Y_val), dtype = tf.float32), Y_val_pred)
        
    
    return [weights_list, list(err.squeeze()), Y_val_pred.numpy()]


def get_data():
    X = np.array([[0.685938, -0.5756752], [0.944493, -0.02803439], [0.9477775, 0.59988844], [0.20710745, -0.12665261], [-0.08198895, 0.22326154], [-0.77471393, -0.73122877], [-0.18502127, 0.32624513], [-0.03133733, -0.17500992], [0.28585237, -0.01097354], [-0.19126464, 0.06222228], [-0.0303282, -0.16023481], [-0.34069192, -0.8288299], [-0.20600465, 0.09318836], [0.29411194, -0.93214977], [-0.7150941, 0.74259764], [0.13344735, 0.17136675], [0.31582892, 1.0810335], [-0.22873795, 0.98337173], [-0.88140666, 0.05909261], [-0.21215424, -0.05584779]], dtype=np.float32)
    y = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    return (X, y)

def get_data_2():
    X = np.array([[0.55824741, 0.8871946, 0.69239914], [0.25242493, 0.77856301, 0.66000716], [0.4443564, 0.1092453, 0.96508663], [0.66679551, 0.49591846, 0.9536062], [0.07967996, 0.61238854, 0.89165257], [0.36541977, 0.02095794, 0.49595849], [0.56918241, 0.45609922, 0.05487656], [0.38711358, 0.02771098, 0.27910454], [0.16556168, 0.9003711, 0.5345797], [0.70774465, 0.5294432, 0.77920751]], dtype=np.float32)
    y = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    return (X, y)



