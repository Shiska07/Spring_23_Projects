import numpy as np
import sys

def get_X_Y_arrays(filename, dtype_x, dtype_y):
    try:
        f = open(filename, 'r')
    except OSError:
        print(f'{filename} could not be opened.\n')
        sys.exit()
        
    # initialize list to store feature and labels for training data
    features = []             
    labels = []
    
    with f:
        line = f.readline()
        while line != '':
            # strip newline and outer parenthesis
            line = line.strip('\n')
            line = line.strip('( )')
            
            # extrace label and append to labels list
            single_label = line.split('), ')[-1]
            labels.append(single_label)
            
            # extrace features and append to features list
            feat = line.split('), ')[0].split(', ')
            features.append(feat)
            
            # read next line
            line = f.readline()
        
        # create dataframe of features and append labels
        X = np.array(features, dtype = dtype_x, ndmin = 2)
        
        # convert labels list to array
        Y = np.array(labels, dtype = dtype_y, ndmin = 2)
        
        return X.transpose(), Y
    
def get_X_array(filename, dtype_x):
    try:
        f = open(filename, 'r')
    except OSError:
        print(f'{filename} could not be opened.\n')
        sys.exit()
        
    # initialize list to store feature and labels for training data
    features = []             
    
    with f:
        line = f.readline()
        while line != '':
            
            # get feature values
            line = line.strip('\n')
            line = line.strip('( )')
            feat = line.split(', ')
            features.append(feat)
            
            # read next line
            line = f.readline()
        
        # create dataframe of features and append labels
        X = np.array(features, dtype = dtype_x, ndmin = 2)
        
        return X.transpose()



# compares two arrays and returns accuracy
def get_acc(Y_pred, Y):
    n_samples = Y_pred.shape[1]
    return (np.sum(Y_eval == Y_pred))/n_samples

# adds bias as the first row to a dataset
def add_bias(X):
    
    n_feat, n_samples = X.shape
    X_b = np.ones((n_feat+1, n_samples), dtype = float)
    X_b[1::,:] = X
    
    return X_b

# given an array of attribute values for a categocial attribute,
# preforms one-hot-encoding and returns resulting binary array
def one_hot_encoder(arr):
    
    __, n_samples = arr.shape
    
    # get unique labels
    uniq_labels = set(arr[0,:].tolist())
    
    # get number of total attribute values
    n_labels = len(uniq_labels)
    
    # create an array of size n_labels*n_samples to store encoded values
    encoded_arr = np.zeros((n_labels, n_samples), dtype = int)
    
    # create dictionary to store row indev of each attribute value
    label_idx_dict = {}
    for i, v in enumerate(uniq_labels):
        label_idx_dict[v] = i
        
    # fill encoded_arr using attribute index dictionary and input arr
    for i in range(n_samples):
        # get index to encode as 1
        idx = label_idx_dict[arr[0,i]]
        encoded_arr[idx, i] = 1
        
    return encoded_arr, label_idx_dict

# given a vector of parobaility values, returns label with max probability for a single sample
def get_sample_prediction_label(sfmax_net, label_idx_dict):
    
    # get inverse of the dictionary
    inv_label_idx_dict = {v: k for k, v in label_idx_dict.items()}
    
    # return label with max probability value
    return inv_label_idx_dict[int(np.argmax(sfmax_net, axis = 0))]


# uses softmax function and parameter matrix to get probability values
# for multiclass classification
def get_sample_prediction_values(x_sample, model_params):
    
    # calculate linear net value
    net = np.dot(model_params, x_sample)
    
    # calculate exponential value for rach class
    exp_net = np.exp(net, dtype = float)
    
    # calculate softmax value for each class
    sfmax_net = exp_net/np.sum(exp_net, axis = 0, dtype = float)
    
    return sfmax_net

# gets predictions for an entire test dataset
def get_predictions(X_test, model_params, label_idx_dict):

    # initialize list to store predictions
    Y_pred = []

    # get number of test samples
    n_dim, n_samples = X_test.shape

    for i in range(n_samples):
    
        x_sample = X_test[:,i].rechape(n_dim, 1)
        y_pred_values = get_sample_prediction_values(X_test[:,i], model_params)
        y_pred_label = get_sample_prediction_label(y_pred_values, label_idx_dict)
        Y_pred.append(y_pred_label)

    # convert labels list to numpy array
    Y_pred = np.array(Y_pred, dtype = str, ndmin = 2)

    return Y_pred 

# trains a softmax regression model given training data, alpha and number of epochs
def train_softmax_regressor(X_train, Y_train, alpha, epochs):
    
    # get number of features and samples
    n_feat, n_samples = X_train.shape
    
    # get no of classes/labels
    n_class, __ = Y_train.shape

    # get paramater matrix
    model_params = np.random.uniform(0, 0.01, size = (n_class, n_feat))

    # initialize list to store net change in parameter values
    change_model_params = []
    
    for i in range(epochs):

        # initialize gradient vector for each epoch
        gradient_mtx = np.zeros((n_class, n_feat), dtype = float)

        for j in range(n_samples):
            
            # pick a sample 
            x_sample = X_train[:,j].reshape(n_feat,1)
            y_sample = Y_train[:,j].reshape(n_class, 1)
            
            # get prediction value
            y_pred = get_sample_prediction_values(x_sample, model_params)

            # calculate gradient matrix
            sample_gradient = np.dot((y_sample - y_pred), x_sample.transpose())
            gradient_mtx = gradient_mtx + sample_gradient
            
        # adjust parameter values using batch gradient descent 
        updated_params = model_params + (alpha*gradient_mtx)
        
        # get the net change in parameters
        net_change = np.sum(np.abs(model_params - updated_params))
        change_model_params.append(net_change)
            
        # set updated parameters as new parameters  
        model_params = updated_params.copy()
        
    # return final parameter matrix
    return model_params


def leave_one_out_evaluation(X_eval, Y_eval, alpha, epochs):

    # add bias to X
    X_train_b = add_bias(X_eval)

    # get number of features and samples
    n_feat, n_samples = X_eval.shape

    # prediction labels generated by 'predict_class_with_knn' will be stored in this list
    Y_pred = []

    # get encoded array for y
    Y_train_encoded, label_idx_dict = one_hot_encoder(Y_eval)
    n_class, __ = Y_train_encoded.shape
    
    # trains a new model to predict each sample
    for i in range(n_samples):
        
        # pick test datapoint
        x_test = X_train_b[:,i].reshape(n_feat, 1)
        y_test = Y_train_encoded[:,i].reshape(n_class, 1)
        
        # create traiing set by deleting test datapoint
        X_train = np.delete(X_train_b, i, axis = 1)
        Y_train = np.delete(Y_train_encoded, i, axis = 1)
        
        # train model
        model_params = train_softmax_regressor(X_train, Y_train, alpha, epochs)
        
        # get test data prediction
        y_pred_values = get_sample_prediction_values(x_test, model_params)
        y_pred_label = get_sample_prediction_label(y_pred_values, label_idx_dict)
        Y_pred.append(y_pred_label)
    
    # convert prediction list to numpy array
    Y_pred = np.array(Y_pred, dtype = str, ndmin = 2)
    acc = get_acc(Y_pred, Y_eval)
    
    # return predictions and accuracy
    return Y_pred, acc

fname_eval = 'Data/3_eval.txt'

X_eval, Y_eval = get_X_Y_arrays(fname_eval, float, str)

# X_eval = np.delete(X_eval, 3, axis = 0)
# model_params, label_idx_dict = train_softmax_regressor(X_train, Y_train, 0.0001, 1)
Y_pred, acc = leave_one_out_evaluation(X_eval, Y_eval, 0.001, 1000)

# 0.001, 1000


