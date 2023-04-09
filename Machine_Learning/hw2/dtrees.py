import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
        Y = np.array(labels, dtype=dtype_y)

        return X, Y


# splits data into training and test set
# first 6 datapoints become the test datapoints
def split_data(X, Y):

    X_test = X[0:6, :]
    Y_test = Y[0:6]

    X_train = np.delete(X, np.s_[0:6], axis = 0)
    Y_train = np.delete(Y, np.s_[0:6])

    return X_train, Y_train, X_test, Y_test


# given an array of categorical values, returns encoded index
def get_encoded_Y(Y):
    # get number of samples
    n_samp = Y.shape[0]

    # get all unique labels
    uniq_labels = set(Y.tolist())

    # create dictionary to store encodings
    encoding_dict = {}
    for i, label in enumerate(uniq_labels):
        encoding_dict[label] = i

    # get number of labels
    n_labels = len(uniq_labels)

    # create encoded arr
    encoded_arr = np.zeros(n_samp, dtype=np.int16)
    for i in range(n_samp):
        encoded_arr[i] = encoding_dict[Y[i]]

    # reverse dictionary for decoding
    inv_encoding_dict = {v: k for k, v in encoding_dict.items()}

    return encoded_arr, inv_encoding_dict

# READ FILE
fname = '2_data.txt'

# read data
X, Y = get_X_Y_arrays(fname, float, str)

# get encoded Y
encoded_Y, decoding_dict = get_encoded_Y(Y)

# split data into training and test
X_train, Y_train, X_test, Y_test = split_data(X, encoded_Y)

# create dataframe
df_train = pd.DataFrame(np.concatenate((X_train, Y_train.reshape(X_train.shape[0], 1)), axis = 1),
                  columns = ['height', 'diameter', 'weight', 'hue', 'labels'], dtype = np.float32)

df_test = pd.DataFrame(np.concatenate((X_test, Y_test.reshape(X_test.shape[0], 1)), axis = 1),
                  columns = ['height', 'diameter', 'weight', 'hue', 'labels'], dtype = np.float32)


# given a dataframe and a feature name, sorts the df and calculates thresholds
def get_thresholds(df, feat_name):

    # sort dataframe according to the given label
    df_sorted = df.sort_values(feat_name, axis = 0, ascending = True, ignore_index = True)

    n_samp = df_sorted.shape[0]
    thresholds = []
    ser = df_sorted[feat_name]

    for i in range(n_samp - 1):
        val = (ser[i] + ser[i+1])/2
        thresholds.append(val)

    return df_sorted, thresholds


# calculates entropy of left and right child
def entropy(val_count):

    # get total classes and items
    n_classes = val_count.size
    n_samp = np.sum(val_count)

    val_count_non_zero = np.ones(val_count.size)

    for n in range(n_classes):
        if val_count[n] != 0:
            val_count_non_zero[n] = val_count[n]

    # if any of the nodes have 0 items, the netropy will be 0
    if n_samp == 0:
        entropy_val = 0
    else:
        # log(0) can cause error so replace term with 1
        entropy_val = np.dot(val_count / n_samp, np.log(val_count_non_zero / n_samp))

    return entropy_val


# splits a df into two based on a feature's threshold
def split_df(df, feat_name, threshold):

    # split df on threshold
    l_df = df.sort_values(feat_name, ascending=True, ignore_index=True)[df[feat_name] <= threshold]
    r_df = df.sort_values(feat_name, ascending=True, ignore_index=True)[df[feat_name] > threshold]

    return l_df, r_df


# calculates the best threshold, lowest entropy and best attribute to split on given a dataframe
def get_best_threshold(df, feat_name, label_name):

    # get number of classes and samples
    n_classes = df[label_name].nunique()
    n_samp = df.count()[label_name]

    # get sorted df and thresholds
    df_sorted, thresholds = get_thresholds(df, feat_name)

    # number of data items for each class on the left and right side of the treshold
    l_counts = np.zeros(n_classes, dtype = np.int16)
    r_counts =  np.zeros(n_classes, dtype = np.int16)

    # initalize variable to store entropy values
    min_l_entropy = math.log(n_classes, 2)
    min_r_entropy = math.log(n_classes, 2)
    min_entropy = (min_l_entropy + min_r_entropy)/2
    best_threshold = thresholds[0]

    # for each threshold
    for thres_val in thresholds:
        # for each class
        for c in range(n_classes):
            l_counts[c] = df_sorted[(df_sorted[label_name] == c) &
                                           (df_sorted[feat_name] <= thres_val)]['feat_name'].count()
            r_counts[c] = df_sorted[(df_sorted[label_name] == c) &
                                           (df_sorted[feat_name] > thres_val)]['feat_name'].count()

        l_entropy = entropy(l_counts)
        r_entropy = entropy(r_counts)
        weighted_entropy = (np.sum(l_counts)*l_entropy + np.sum(r_counts)*r_entropy)/n_samp
        if weighted_entropy < min_entropy:
            min_entropy = weighted_entropy
            min_l_entropy = l_entropy
            min_r_entropy = r_entropy
            best_threshold = thres_val

    return best_threshold, min_entropy, min_l_entropy, min_r_entropy

# calculates the best attribute to split a node on
def get_best_split(df, n_classes, entropy_p, label_name):
    # create a list of all attributes
    feat_list = df.columns[:-1].tolist()
    best_feat_idx = 0                       # index of best feature to split on

    l_entropy = math.log(n_classes, 1)
    r_entropy = math.log(n_classes, 1)
    min_entropy = (l_entropy + r_entropy)/2

    # store results for all features in a list
    entropy_calculations = []
    for feat in feat_list:
        entropy_calculations.append([get_best_threshold(df, feat, label_name)])

    # get feature with minimum split entropy
    for i in range(len(feat_list)):
        if entropy_calculations[i][1] < min_entropy:
            min_entropy = entropy_calculations[i][1]
            best_feat_idx = i

    # calculate information gain and store entropy values
    info_gain = entropy_p - min_entropy
    best_feat = feat_list[best_feat_idx]
    best_thresh = entropy_calculations[best_feat_idx][0]
    l_entropy = entropy_calculations[best_feat_idx][2]
    r_entropy = entropy_calculations[best_feat_idx][3]

    return best_feat, best_thresh, info_gain, l_entropy, r_entropy


# returns the label of a leaf node given Y labels
def get_leaf_value(Y_ser):

    # return label with the highest count
    most_count = Y_ser.mode()[0]

    return decoding_dict[most_count]


class Node:
    def __int__(self, feat_name=None, threshold=None, node_entropy=None, l_node = None, r_node = None, class_label = None):

        self.feat_name = feat_name        # splitting feature
        self.threshold = threshold        # splitting threshold
        self.entropy = node_entropy       # entropy value of the node
        self.l_node = l_node              # left child node
        self.r_node = r_node              # left child node
        self.class_label = class_label    # class label if a left node, None means not a leaf node


class DTreeClassifier:
    def __init__(self, max_depth = None, min_samp_to_split = 2):

        # major attributes are root, mx_depth and min samples required for splitting
        self.root = None
        self.max_depth = max_depth
        self.min_samp_to_split = min_samp_to_split

    # recursively builds a descision tree classifier
    def build_tree(self, df, label_name, curr_depth = 0):

        # get total number of classes and samples
        n_classes = df[label_name].nunique()
        n_samples = df.count()[label_name]

        # calculate entropy of parent
        p_val_counts = np.zeros(n_classes)
        for c in range(n_classes):
            p_val_counts[c] = df[df[label_name] == c][label_name].count()
        entropy_p = entropy(p_val_counts)

        # only split if these criteria are met
        if n_samples > self.min_samp_to_split and curr_depth < self.max_depth:
            feat_name, threshold, info_gain, l_entropy, r_entropy = get_best_split(df_train, entropy_p, n_classes,
                                                                                   label_name)
            if info_gain > 0:
                l_df, r_df = split_df(df_train, feat_name, threshold)
                l_node = self.build_tree(l_df, label_name, curr_depth + 1)
                r_node = self.build_tree(r_df, label_name, curr_depth + 1)

                return Node(feat_name, threshold, entropy_p, l_node, r_node)

        node_class_label = get_leaf_value(df['label_name'])
        return Node(class_label = node_class_label)


