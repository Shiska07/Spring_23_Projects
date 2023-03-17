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
    
# gets sample squared error
def get_sample_squared_error(y_sample, y_pred):
    
    # calculate and return squared error
    return np.square(y_sample - y_pred)


# gets average squared error for an entire test dataset
def get_avg_squared_error(Y_pred, Y_test):
    
    n_samples = Y_pred.shape[1]
    sum_of_sq_err = np.sum((np.square(Y_pred - Y_test)), axis = 1)
    
    # return average over number of samples
    return sum_of_sq_err[0]/n_samples


# gets prediction value for a sample
def get_y_pred_value_lolr(x_sample, model_params):
    
    y_pred = np.dot(model_params, x_sample)

    # return prediction value as a scalar
    return y_pred[0, 0]

# transform training data given gaussian weights
def transform_data(X_train, Y_train, x_point, gamma):

    # # get number of training samples
    n_dim, n_train_samples = X_train.shape

    #initialize array to store weights
    weights = np.zeros((1, n_train_samples), dtype = float)
    
    d = float(-2*gamma*gamma)
    for i in range(n_train_samples):
        weights[:,i] = np.exp(np.dot((X_train[:,i].reshape(n_dim,1) - x_point).transpose(),
                                     (X_train[:,i].reshape(n_dim,1) - x_point))/d)

    # transform data
    X_trans = weights*X_train
    Y_trans = weights*Y_train

    return X_trans, Y_trans

# trains a locally weighted linear regression model given a point 'x_point'
def train_model_lolr(X_train, Y_train, x_point, epochs, alpha, gamma):
    
    # use datapoint 'x_point' datapoint to transform data
    X_trans, Y_trans = transform_data(X_train, Y_train, x_point, gamma)
    
    # get number of training samples
    n_dim, n_train_samples = X_trans.shape

    # get output dimension
    n_out, __ = Y_train.shape
    
    # initialize model parameters
    model_params = np.random.uniform(0, 0.01, size = (n_out, n_dim))
    
    # do this per epoch
    for i in range(epochs): 
        
        # initialize gradient vector for each epoch
        gradient_vec = np.zeros((n_out, n_dim), dtype = float)
        
        # for every training sample
        for j in range(n_train_samples):
             
            x_sample = X_trans[:,j].reshape(n_dim, 1)
            y_sample = Y_trans[:,j][0]
            
            # get prediction value and adjust weights
            y_pred = get_y_pred_value_lolr(x_sample, model_params)
            
            # sum gradients over all training samples
            gradient_vec = gradient_vec + ((y_pred - y_sample)*x_sample.transpose())
            
        # adjust parameter values using batch gradient descent 
        model_params = model_params - (alpha*gradient_vec)
        
    # return final parameter vector
    return model_params


# returns predicted values for an entire test dataset
def get_predictions_lolr(X_train, Y_train, X_test, epochs, alpha, gamma):
    
    # save number of test samples
    n_feat, n_train_samples = X_train.shape
    __, n_test_samples = X_test.shape

    # add bias to X data
    X_train_b = np.ones((n_feat+1, n_train_samples))
    X_train_b[1,:] = X_train
    X_test_b = np.ones((n_feat+1, n_test_samples))
    X_test_b[1,:] = X_test
    
    # store prediction values
    predictions = []

    # for every test sample
    for i in range(n_test_samples):

        # get test sample to train model om
        x_test_sample = X_test_b[:, i].reshape(n_feat+1, 1)
        
        # train model for local parameters
        model_params = train_model_lolr(X_train_b, Y_train, x_test_sample, epochs, alpha, gamma)
        y_sample_pred = get_y_pred_value_lolr(x_test_sample, model_params)
        predictions.append(y_sample_pred)
    
    # convert list to array
    Y_pred = np.array(predictions, dtype = float, ndmin = 2)
    
    return Y_pred

# get training data
fname_train = "Data/1_b_c_2_train.txt"
X_train, Y_train = get_X_Y_arrays(fname_train, float, float)

# set paramters
gamma = 0.13
alpha = 0.01
epochs = 200
n_train_samples = X_train.shape[1]

# create range for plotting
X_range = (np.linspace(np.amin(X_train), np.amax(X_train), n_train_samples)).reshape(1, n_train_samples)

# get predictions for x_range
Y_range_pred = get_predictions_lolr(X_train, Y_train, X_range, epochs, alpha, gamma)
