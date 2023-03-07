import numpy as np
import sys

# given a file
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
    
# k = frequency increment
# d = function depth
# given input datapoint 'x_sample', 
# returns transformed version of the intput datapoint as a numpy array
def get_feature_vector(x_sample, k, d):
    
    # stored transformed values in a list
    trans_feat_list = []
    
    # append 1 and value of 'x_sample' to the list
    trans_feat_list.append(float(1))
    trans_feat_list.append(float(x_sample))
    
    # remaining transformations will be based on 'k' and 'd'
    for i in range(1, d+1):
        val1 = (np.sin(i*k*x_sample, dtype = float)**(i*k))*np.cos(x_sample, dtype = float)
        trans_feat_list.append(val1)
        val2 = (np.cos(i*k*x_sample, dtype = float)**(i*k))*np.sin(x_sample, dtype = float)
        trans_feat_list.append(val2)
    
    # convert list into array
    x_sample_trans = np.array(trans_feat_list, dtype = float, ndmin = 2).transpose()

    # return transformed features
    return x_sample_trans


# calculates sample squared error
def get_sample_squared_error(y_sample, y_pred):
    
    # calculate and return squared error
    return np.square(y_sample - y_pred)


# get sum of squared error for an entire test dataset
def get_sum_of_squared_error(Y_pred, Y_test):

    sum_of_sq_err = np.sum((Y_pred - Y_test)**2, axis = 1)

    return sum_of_sq_err


# get's prediction value for a sample
def get_prediction_value(x_sample, model_params, k, d):
    
    y_pred = np.dot(model_params, get_feature_vector(x_sample, k, d))

    # return prediction value as a scalar
    return y_pred[0, 0]


# returns predicted values and squared error given test data
def get_prediction(X_test, model_params, k, d):
    
    # save number of test samples
    n_feat, n_test_samples = X_test.shape
    
    # initialize list to store prediction values
    predictions = []
    
    for i in range(n_test_samples):
        y_pred = get_prediction_value(X_test[:,i][0], model_params, k, d)
        predictions.append(y_pred)

    Y_pred = np.array(predictions, dtype = float, ndmin = 2)
        
    return Y_pred


# trains a linear regression model 
def train_model(X_train, Y_train, epochs, alpha, k, d):
    
    # get number of training samples
    n_feat, n_samples = X_train.shape
    
    # get output dimension
    n_out, __ = Y_train.shape
    
    # initialize parameter vector
    model_params = np.random.randn(n_out, (2*d)+2)
    gradient_vec = np.zeros((n_out, (2*d)+2), dtype = float)

    # do this per epoch
    for i in range(epochs):    
        for j in range(n_samples):
            
            # pick a sample randomly
            idx = np.random.randint(0, n_samples)
            x_sample = X_train[:,idx]
            y_sample = Y_train[:,idx]
            
            # get prediction value and adjust weights
            y_pred = get_prediction_value(x_sample[0], model_params, k, d)
            gradient_vec = gradient_vec + (y_pred - y_sample[0])*(get_feature_vector(x_sample[0], k, d).transpose())
            
        # adjust parameter values using batch gradient descent   
        model_params = model_params - (alpha*gradient_vec)
    # return final parameter vector
    return model_params


k = 8
epochs = 5
alpha = 0.00001
fname_train = '1_b_c_2_train.txt'
fname_test = '1_c_test.txt'


X_train, Y_train = get_X_Y_arrays(fname_train, float, float)
X_test, Y_test = get_X_Y_arrays(fname_test, float, float)

# store parameters for madels and errors for different functional depths
model_params_list = []
squared_error_list = []

for d in range(0, 7):
    model_params = train_model(X_train, Y_train, epochs, alpha, k, d)
    model_params_list.append(model_params)
    Y_pred = get_prediction(X_test, model_params, k, d)
    squared_error = get_sum_of_squared_error(Y_pred, Y_test)
    squared_error_list.append(squared_error)

print(squared_error_list)
