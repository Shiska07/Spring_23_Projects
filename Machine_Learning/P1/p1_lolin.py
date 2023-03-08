import numpy as np

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

# calculates sample squared error
def get_sample_squared_error(y_sample, y_pred):
    
    # calculate and return squared error
    return np.square(y_sample - y_pred)


# get average squared error for an entire test dataset
def get_avg_squared_error(Y_pred, Y_test):

    sum_of_sq_err = np.sum((Y_pred - Y_test)**2, axis = 1)
    
    # return average over number of samples
    return sum_of_sq_err[0]/Y_pred.shape[1]


# get's prediction value for a sample
def get_y_pred_value_lolr(x_sample, model_params):
    
    y_pred = np.dot(model_params, x_sample)

    # return prediction value as a scalar
    return y_pred[0, 0]


# transform training data given gaussian weights
def transform_data(X_train, Y_train, x_point, gamma):

    # # get number of training samples
    n_feat, n_train_samples = X_train.shape

    #initialize array to store weights
    weights = np.zeros((n_feat, n_train_samples), dtype = float)

    for i in range(n_train_samples):
        weights[:,i] = float(np.exp(-((X_train[:,i] - x_point[1])**2)/(2*gamma**2)))    

    # transform data
    X_trans = weights*X_train
    Y_trans = weights*Y_train

    return X_trans, Y_trans

# trains a linear regression model 
def train_model_lolr(X_train, Y_train, x_point, epochs, alpha, gamma):
    
    # use sample datapoint to transform data
    X_trans, Y_trans = transform_data(X_train, Y_train, x_point, gamma)
    
    # get number of training samples
    n_feat, n_train_samples = X_trans.shape
    
    # add bias to X
    X_train_b = np.ones((n_feat+1, n_train_samples), dtype = float)
    X_train_b[1,:] = X_trans

    # get output dimension
    n_out, __ = Y_train.shape
    
    # initialize parameter vector
    model_params = np.random.uniform(-1, 1, size = (n_out, n_feat+1))

    # store net change in parameter values
    change_model_params = []

    # store error over epoch
    err_over_epoch_single_point = []
    
    # do this per epoch
    for i in range(epochs): 
        
        # initialize gradient vector for each epoch
        gradient_vec = np.zeros((n_out, n_feat+1), dtype = float)
        
        for j in range(n_train_samples):
            
            # pick a sample 
            x_sample = X_train_b[:,j].reshape(n_feat+1, 1)
            y_sample = Y_trans[:,j]
            
            # get prediction value and adjust weights
            y_pred = get_y_pred_value_lolr(x_sample, model_params)
            
            # sum gradients over all training samples
            gradient_vec = gradient_vec + (y_pred - y_sample[0])*(x_sample.transpose())
            
        # adjust parameter values using batch gradient descent 
        updated_params = model_params - (alpha*gradient_vec)
        
        # get the net change in parameters
        net_change = np.sum(np.abs(model_params - updated_params))
        change_model_params.append(net_change)
    
        # set updated parameters as new parameters  
        model_params = updated_params.copy()

        
    # return final parameter vector
    return model_params


# returns predicted values for a given test data
def get_predictions_lolr(X_train, Y_train, X_test, Y_test, epochs, alpha, gamma):
    
    # save number of test samples
    n_feat, n_test_samples = X_test.shape

    # add bias to test data
    X_test_b = np.ones((n_feat+1, n_test_samples))
    X_test_b[1,:] = X_test
    
    # store prediction values
    predictions = []

    
    for i in range(n_test_samples):

        # get test sample to train model om
        x_test_sample = X_test_b[:, i].reshape(n_feat+1, 1)
        
        # train model for local parameters
        model_params = train_model_lolr(X_train, Y_train, x_test_sample, epochs, alpha, gamma)
        y_sample_pred = get_y_pred_value_lolr(x_test_sample, model_params)
        predictions.append(y_sample_pred)

    Y_pred = np.array(predictions, dtype = float, ndmin = 2)

    avg_squared_err = get_avg_squared_error(Y_pred, Y_test)
        
    return Y_pred, avg_squared_err

# training and test filenames
fname_train = "Data/1_b_c_2_train.txt"
fname_test = "Data/1_c_test.txt"

# get data
X_train, Y_train = get_X_Y_arrays(fname_train, float, float)
X_test, Y_test = get_X_Y_arrays(fname_test, float, float)
gamma = 0.13
alpha = 0.001
epochs = 50

training_err_over_epochs = []

for epoch in range(1, epochs+1):
    Y_train_pred, avg_train_squared_err = get_predictions_lolr(X_train, Y_train, X_train, Y_train, epoch, alpha, gamma)
    training_err_over_epochs.append(avg_train_squared_err)  

Y_test_pred, avg_test_squared_err = get_predictions_lolr(X_train, Y_train, X_test, Y_test, epochs, alpha, gamma)
