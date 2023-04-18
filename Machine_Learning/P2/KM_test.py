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
        Y = np.array(labels, dtype=dtype_y)

        return X, Y


fname_train = 'data.txt'

X_train, Y_train = get_X_Y_arrays(fname_train, float, str)


# randomly initializes 'k' distinct centroids given data 'X'
# centroids lie within the range of datapoints
def initialize_cluster_centroids(X, k):

    # get number of features
    n_feat = X.shape[1]

    # create dictionary to be able to number centroids
    centroids_dict = {}

    for i in range(k):

        # pick a random point
        random_idx = np.random.randint(0, X.shape[0])

        centroids_dict[i] = X[random_idx].reshape(1, n_feat)

    return centroids_dict


# given datapoints and cluster centroids, returns array with key
# of the closest cluster
def get_cluster_assignments(X, centroids_dict):

    # get number of centroids
    n_c = len(centroids_dict.keys())
    n_feat = X.shape[1]

    # get number of samples
    n_samp = X.shape[0]

    # initialize arr to store distance from each centroid
    dist_arr = np.zeros((n_samp, n_c))

    # calculate distance from each centroid
    for i in range(n_c):
        # get distance from eaach centroid
        dist_arr[:, i] = np.sqrt(np.sum((X - centroids_dict[i]) ** 2, axis=1))

    # get index of the lowest distance over rows(diff clusters)
    # the indx corresponds to the cluster key
    cluster_assignments = np.argmin(dist_arr, axis=1)

    return cluster_assignments


# given datapoints and number of the closeast cluster,
# computes new centroids
def compute_cluster_centroids(X, cluster_assignments):

    # initalize dict to store centroids
    centroids_dict = {}

    n_samp, n_feat = X.shape

    # get number of clusters
    n = np.max(cluster_assignments) + 1

    for i in range(n):

        # datapoints belonging to cluster i
        x_i = X[cluster_assignments == i]
        centroids_dict[i] = np.sum(x_i, axis=0) / x_i.shape[0]

    return centroids_dict


# given datapoints, assigned_cluster and centroids, computes the
# average squared distance of the clusters
def get_average_squared_distance(X, cluster_assignments, centroids_dict):

    n_samp, n_feat = X.shape

    # get number of clusters
    n_c = np.max(cluster_assignments) + 1

    # stores average squared distance
    sum_sq_dist = 0

    # for each cluster
    for i in range(n_c):
        # calculate distance between datapoints belonging to cluster i
        # with centroid of cluster i
        dist_arr = np.sum((X[cluster_assignments == i] - centroids_dict[i]) ** 2, axis=1)

        # calculate sum of squared distances
        sum_sq_dist = sum_sq_dist + np.sum(dist_arr)

    return sum_sq_dist / n_c


def get_predictions(X_train, Y_train, cluster_assignments):

    n_samp, n_feat = X_train.shape

    # get number of clusters
    n_c = np.max(cluster_assignments) + 1

    # create array to store predictions
    Y_pred = np.zeros(n_samp, dtype=object)

    # create dict to store majority label for the cluster
    cluster_labels = {}

    # concatenate X with cluster assignments and actual labels
    # to assign predicted labels
    X = np.concatenate((X_train, Y_train.reshape(n_samp,1)), axis = 1)

    for i in range(n_c):

        # get actual labels for all datapoints assigned to cluster number i
        labels = X[cluster_assignments == i][:, -1]

        # get count of each label
        values, counts = np.unique(labels, return_counts=True)

        # index of label with the highest count
        mode_idx = np.argmax(counts)

        # label with the highest count
        mode = values[mode_idx]

        # save label of the given cluster
        cluster_labels[i] = mode

    for i in range(n_samp):

        # get predicted labels from each sampple according to the
        # cluster it belongs to
        Y_pred[i] = cluster_labels[cluster_assignments[i]]

    return Y_pred


# returns weighted accuracy
def get_weighted_acc(Y_pred, Y_train, cluster_assignments):

    # get number of samples and clusters
    n_samp = Y_pred.shape[0]
    n_c = np.max(cluster_assignments) + 1

    # concat values for easy computation
    Y = np.concatenate((Y_train.reshape(n_samp, 1), Y_pred.reshape(n_samp, 1), cluster_assignments.reshape(n_samp, 1)), axis=1)

    # initalize array to store weights and accuracy
    weights = np.zeros((1, n_c))
    acc = np.zeros((1, n_c))

    for i in range(n_c):
        # all prediction values belonging to cluster i
        y_i = Y[cluster_assignments == i]

        # weight for cluster i
        weights[0, i] = y_i.shape[0] / n_samp

        # accuracy for cluster i
        acc[0, i] = np.sum((y_i[:, 0] == y_i[:, 1])) / y_i.shape[0]

    # caluclate the weighted sum
    weighted_acc = np.sum(weights * acc)

    return weighted_acc


# given an unlabeled data, returns k number of clusters
def get_kmeans_clusters(X, k):

    # initalize cluster centroids
    cluster_centroids_dict = initialize_cluster_centroids(X, k)

    # initialize list to store average squared distance
    avg_sq_dist_list = []

    # get new cluster assignments
    prev_cluster_assignments = get_cluster_assignments(X, cluster_centroids_dict)

    while True:

        # compute centroids
        cluster_centroids_dict = compute_cluster_centroids(X, prev_cluster_assignments)

        # add average squared distance
        avg_sq_dist_list.append(get_average_squared_distance(X, prev_cluster_assignments, cluster_centroids_dict))

        # get new cluster assignments
        cluster_assignments = get_cluster_assignments(X, cluster_centroids_dict)

        # if new assignments is same as the old assignments, stop
        if (prev_cluster_assignments == cluster_assignments).all():
            break

        # save new assignments
        prev_cluster_assignments = cluster_assignments

    return cluster_assignments

final_cluster_assignments = get_kmeans_clusters(X_train, 3)
Y_pred = get_predictions(X_train, Y_train, final_cluster_assignments)
acc = get_weighted_acc(Y_pred, Y_train, final_cluster_assignments)
