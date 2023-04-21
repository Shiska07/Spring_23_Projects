import numpy as np
import sys
import matplotlib.pyplot as plt


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
        X = np.array(features, dtype=dtype_x, ndmin=2)

        # convert labels list to array
        Y = np.array(labels, dtype=dtype_y, ndmin=2)

        return X.transpose(), Y


# given an array of attribute values for a categocial attribute,
# preforms one-hot-encoding and returns resulting binary array
def one_hot_encoder(arr):
    __, n_samples = arr.shape

    # get unique labels
    uniq_labels = set(arr[0, :].tolist())

    # get number of total attribute values
    n_labels = len(uniq_labels)

    # create an array of size n_labels*n_samples to store encoded values
    encoded_arr = np.zeros((n_labels, n_samples), dtype=int)

    # create dictionary to store row indev of each attribute value
    encoding_dict = {}
    for i, v in enumerate(uniq_labels):
        encoding_dict[v] = i

    # fill encoded_arr using attribute index dictionary and input arr
    for i in range(n_samples):
        # get index to encode as 1
        idx = encoding_dict[arr[0, i]]
        encoded_arr[idx, i] = 1

    # get inverse of the dictionary
    decoding_dict = {v: k for k, v in encoding_dict.items()}

    return encoded_arr, encoding_dict, decoding_dict


# given a one-hot encoded array and a decoding_dict returns decoded array
def get_decoded_arr(arr, decoding_dict):
    # get number of samples
    n_samples = arr.shape[1]

    arr_decoded = np.zeros((1, n_samples), dtype=object)

    for i in range(n_samples):
        arr_decoded[:, i] = decoding_dict[np.argmax(arr[:, i])]

    return arr_decoded


# given an array of labels and encoding dict returns encoded array
def get_encoded_arr(arr, encoding_dict):
    # get number of classes and number of samples
    n_class = len(encoding_dict.keys())
    n_samples = arr.shape[1]

    encoded_arr = np.zeros((n_class, n_samples), dtype=int)

    for i in range(n_samples):
        idx = encoding_dict[arr[:, i][0]]
        encoded_arr[idx, i] = int(1)

    return encoded_arr


# adds bias as the first row to a dataset
def add_bias(X):
    n_feat, n_samples = X.shape
    X_b = np.ones((n_feat + 1, n_samples), dtype=float)
    X_b[1::, :] = X

    return X_b


# plots a line graph
def plot_train_vs_test_acc(train, test, x, title, xlab, ylab):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(x, train, color='blue')
    plt.plot(x, test, color='green')

    # add title and labels
    plt.title(title, fontdict={'fontsize': 14})
    plt.xlabel(xlab, fontdict={'fontsize': 12})
    plt.ylabel(ylab, fontdict={'fontsize': 12})
    plt.xlim(0, max(x) + 5)
    plt.legend(['training acc', 'test, acc'])
    plt.grid()


# compares two arrays and returns class accuracy
# Y_pred.shape == Y.shape = ndim, n_samples
# Y_pred.dtype = Y.dtype + string object
def get_class_acc(Y_pred, Y):
    # create a dict to store class accuracy
    class_acc = {}

    # get all unique classes
    classes = set(Y[0, :].tolist())

    # get number of samples
    n_samples = Y_pred.shape[1]

    # calculate total occurence and accurate predictions for each class
    for c in classes:

        total = 0
        acc_vals = 0

        for i in range(n_samples):
            if Y[:, i] == c:
                total = total + 1
                if Y[:, i] == Y_pred[:, i]:
                    acc_vals = acc_vals + 1

        class_acc[c] = acc_vals / total

    return class_acc


# compares two arrays and returns overall accuracy
# Y_pred.shape == Y.shape = ndim, n_samples
# Y_pred.dtype = Y.dtype + string object
def get_acc(Y_pred, Y):
    n_samples = Y_pred.shape[1]

    # reutrn overall accuracy
    return (np.sum(Y == Y_pred)) / n_samples


# given a vector of parobaility values, returns label with max probability for a single sample
def get_sample_prediction_label(sfmax_net, decoding_dict):
    # return label with max probability value
    return decoding_dict[int(np.argmax(sfmax_net, axis=0))]


# uses softmax function and parameter matrix to get probability values
# for multiclass classification
def get_sample_prediction_values(x_sample, model_params):
    # calculate linear net value
    net = np.dot(model_params, x_sample)

    # calculate exponential value for rach class
    exp_net = np.exp(net, dtype=float)

    # calculate softmax value for each class
    sfmax_net = exp_net / np.sum(exp_net, axis=0, dtype=float)

    return sfmax_net


# gets predictions for an entire test dataset
# returns predictions as labels
def get_predictions(X_test, model_params, decoding_dict):
    # initialize list to store predictions
    Y_pred = []

    # get number of test samples
    n_feat, n_samples = X_test.shape

    for i in range(n_samples):
        x_sample = X_test[:, i].reshape(n_feat, 1)
        y_pred_values = get_sample_prediction_values(X_test[:, i], model_params)
        y_pred_label = get_sample_prediction_label(y_pred_values, decoding_dict)
        Y_pred.append(y_pred_label)

    # convert labels list to numpy array
    Y_pred = np.array(Y_pred, dtype=str, ndmin=2)

    return Y_pred


# training with batch gradient descent
def train_softmax_regressor_batch(X_train, Y_train, weights_arr, alpha, epochs):
    # get number of features and samples
    n_feat, n_samples = X_train.shape

    # get no of classes/labelsexit
    n_class, __ = Y_train.shape

    # get paramater matrix
    model_params = np.random.uniform(-0.01, 0.01, size=(n_class, n_feat))

    # initialize list to store net change in parameter values
    epoch_change_model_params = []

    for i in range(epochs):

        # initialize gradient vector for each epoch
        gradient_mtx = np.zeros((n_class, n_feat), dtype=float)

        for j in range(n_samples):
            # pick a sample
            x_sample = X_train[:, j].reshape(n_feat, 1)
            y_sample = Y_train[:, j].reshape(n_class, 1)

            # get prediction value
            y_pred = get_sample_prediction_values(x_sample, model_params)

            # calculate gradient matrix
            sample_gradient = (weights_arr[:, j][0]) * np.dot((y_sample - y_pred), x_sample.transpose())

            # sample_gradient = np.dot((y_sample - y_pred), x_sample.transpose())
            gradient_mtx = gradient_mtx + sample_gradient

        # adjust parameter values using batch gradient descent
        updated_params = model_params + (alpha * gradient_mtx)

        # get the net change in parameters
        net_change = np.sum(np.abs(model_params - updated_params))
        epoch_change_model_params.append(net_change)

        # set updated parameters as new parameters
        model_params = updated_params.copy()

    # return final parameter matrix
    return model_params, epoch_change_model_params


fname = 'data.txt'

X, Y = get_X_Y_arrays(fname, float, str)

# add bias to X training data
X_b = add_bias(X)

# get encoded values for Y_train
Y_encoded, encoding_dict, decoding_dict = one_hot_encoder(Y)

def split_data(X, Y, n_test_samples):

    X_test = X[:, 0:n_test_samples]
    Y_test = Y[:, 0:n_test_samples]

    X_train = np.delete(X, np.s_[0:n_test_samples], axis = 1)
    Y_train = np.delete(Y, np.s_[0:n_test_samples], axis = 1)

    return X_train, Y_train, X_test, Y_test

# split data into training and test set
X_train, Y_train, X_test, Y_test = split_data(X_b, Y_encoded, 15)


# calculates new sample weights from old weight values
# Y_pred and Y_true are ont-hot-encoded, alpha is the previous model's weight
def get_data_weights(weights_arr, Y_pred_encoded, Y_train_encoded, alpha):
    # get labels
    Y_pred_labels = get_decoded_arr(Y_pred_encoded, decoding_dict)
    Y_train_labels = get_decoded_arr(Y_train_encoded, decoding_dict)

    truth_vals = (Y_pred_labels == Y_train_labels).astype(int)

    for i in range(truth_vals.shape[1]):
        # set value = -1 everywhere where predictions don't match
        if truth_vals[:, i] == 0:
            truth_vals[:, i] = int(-1)

    new_weights = weights_arr * np.exp(-alpha * truth_vals)

    return new_weights


# calculates weight of the model acoording to accuracy
def get_model_weight(error, n_classes):
    # compute weight
    model_weight = 0.5 * np.log(2*(1 - error) / error)

    return model_weight


def get_weighted_error(Y_pred_encoded, Y_train_encoded, weights_arr, decoding_dict):
    # get labels
    Y_pred_labels = get_decoded_arr(Y_pred_encoded, decoding_dict)
    Y_train_labels = get_decoded_arr(Y_train_encoded, decoding_dict)

    error = np.sum(weights_arr * (Y_pred_labels != Y_train_labels), axis=1) / np.sum(weights_arr, axis=1)

    return error[0]


# calculates null error for the data according to class distribution
def get_null_error(Y_train, n_classes, n_samples, decoding_dict):
    # get labels
    Y_train_labels = get_decoded_arr(Y_train, decoding_dict)

    # get count of each label
    value, counts = np.unique(Y_train_labels, return_counts=True)

    # get max count value
    max_count = max(counts)

    # percent of data belonging to biggest class
    null_acc = max_count / n_samples

    # null error = 1 - null_acc
    return 1 - null_acc


# applying boosting and returns final model parameters and weights for all classifiers
def apply_boosting(X_train, Y_train, n_classifiers, epochs, alpha, decoding_dict, encoding_dict):
    # get number of samples and classes
    n_samples = X_train.shape[1]
    n_classes = len(decoding_dict.keys())

    # initialize array to store weights for each data item
    weights_arr = np.ones((1, n_samples), dtype=float) * (1 / n_samples)

    # initialize list to store model params and vote_weight
    model_params_list = []
    model_weights_list = []

    # get null error value
    null_err = get_null_error(Y_train, n_classes, n_samples, decoding_dict)

    for i in range(n_classifiers):

        # train model and get predictions
        model_params, __ = train_softmax_regressor_batch(X_train, Y_train, weights_arr, alpha, epochs)

        # get prediction on training data
        Y_pred_labels = get_predictions(X_train, model_params, decoding_dict)
        Y_train_labels = get_decoded_arr(Y_train, decoding_dict)

        # get encoded predictions
        Y_pred_encoded = get_encoded_arr(Y_pred_labels, encoding_dict)

        # get weighted error
        error = get_weighted_error(Y_pred_encoded, Y_train, weights_arr, decoding_dict)

        # if the model does better than random
        if error < null_err:

            # add parameters to list
            model_params_list.append(model_params)

            # calculate and save model weight
            model_weight = get_model_weight(error, n_classes)
            model_weights_list.append(model_weight)

            # calculate datapoint weights for next classifier
            weights_arr = get_data_weights(weights_arr, Y_pred_encoded, Y_train, model_weight)

    model_weights_arr = np.array(model_weights_list, dtype=float, ndmin=2)

    return model_params_list, model_weights_arr


# returns prediction on test data according to model weights
def get_ensemble_prediction(X_test, model_params_list, model_weights_arr, decoding_dict, encoding_dict):

    # get number of classifiers, samples and classes
    n_classifiers = len(model_params_list)
    n_samples = X_test.shape[1]
    n_classes = len(decoding_dict.keys())

    # create a list to store predictions for all classifiers
    predictions_list = []

    # create array to store weighted prediction from all classifiers
    class_pred_wts = np.zeros((n_classes, n_samples), dtype=float)

    # get predictions for each classifier
    for i in range(n_classifiers):
        # get labels and convert to one-hot
        Y_pred_labels = get_predictions(X_test, model_params_list[i], decoding_dict)
        Y_pred_encoded = get_encoded_arr(Y_pred_labels, encoding_dict)

        predictions_list.append(Y_pred_encoded)

    # create array of n_samples for column sclicing
    idx_columns = np.arange(0, n_samples)

    # calculate weighted vote for each prediction
    for i in range(n_classifiers):
        # class indices of predicted values
        idx_pred_class = np.argmax(predictions_list[i], axis=0)

        # add weight to label indices that were predicted
        class_pred_wts[idx_pred_class, idx_columns] = class_pred_wts[idx_pred_class, idx_columns] \
                                                      + model_weights_arr[:, i]

    # final predcition is index with the highest weight along rows
    final_predictions = np.argmax(class_pred_wts, axis = 0)

    # get final labels
    final_labels = np.zeros((1, n_samples), dtype = object)

    for i in range(n_samples):
        final_labels[:, i] = decoding_dict[final_predictions[i]]

    return final_labels

epochs = 4000
alpha = 0.1
n_classifiers = 1

model_params, __ = apply_boosting(X_train, Y_train, n_classifiers, epochs, alpha, decoding_dict, encoding_dict)

model_params_list, model_weights = apply_boosting(X_train, Y_train, 10, 5000, 0.1, decoding_dict, encoding_dict)

Y_test_pred = get_ensemble_prediction(X_test, model_params_list, model_weights, decoding_dict, encoding_dict)

