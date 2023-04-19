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
        X = np.array(features, dtype=dtype_x, ndmin=2)     # X.shape = ndim, n_samples

        # convert labels list to array
        Y = np.array(labels, dtype=dtype_y, ndmin=2)       # Y.shape = ndim, n_samples

        return X, Y


fname = 'data.txt'

X, Y = get_X_Y_arrays(fname, float, str)


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

    arr_decoded = np.zeros((1, n_samples), dtype = object)

    for i in range(n_samples):
        arr_decoded[:, i] = decoding_dict[np.argmax(arr[:, i])]

    return arr_decoded


# given an array of labels and encoding dict returns encoded array
def get_encoded_arr(arr, encoding_dict):

    # get number of classes and number of samples
    n_class = len(encoding_dict.keys())
    n_samples = arr.shape[1]

    encoded_arr = np.zeros((n_class, n_samples), dtype = int)

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
def plot_change(X, title, xlab, ylab):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(X, color='red')

    # add title and labels
    plt.title(title, fontdict={'fontsize': 15})
    plt.xlabel(xlab, fontdict={'fontsize': 12})
    plt.ylabel(ylab, fontdict={'fontsize': 12})
    plt.grid()

# add bias to X training data
X_b = add_bias(X)

# get encoded values for Y_train
Y_encoded, encoding_dict, decoding_dict = one_hot_encoder(Y)

decoded_y = get_decoded_arr(Y_encoded[:, 0:5], decoding_dict)
encoded_y = get_encoded_arr(Y[:, 0:5], encoding_dict)

def split_data(X, Y, n_test_samples):

    X_test = X[:, 0:n_test_samples]
    Y_test = Y[:, 0:n_test_samples]

    X_train = np.delete(X, np.s_[0:n_test_samples], axis = 1)
    Y_train = np.delete(Y, np.s_[0:n_test_samples], axis = 1)

    return X_train, Y_train, X_test, Y_test

# split data into training and test set
X_train, Y_train, X_test, Y_test = split_data(X_b, Y_encoded, 6)
