import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = 'UCI_datasets/satellite_training.txt'

# read data
df = pd.read_csv(file_path, sep=' ', lineterminator='\n', header = None)

# drop last column
df.drop([36], axis = 1, inplace = True)

data_arr = df.to_numpy()

# get number of samples and features
n_samp, n_feat = data_arr.shape


# computes centroids given cluster assignments
def calculate_cluster_centroids(cluster_assignments):
    # get number of clusters
    n_c = max(cluster_assignments) + 1

    # dictionary to store new centroids
    centroids_dict = {}

    # compute centroids using cluster assignments
    for cluster_number in range(n_c):

        # use array slicing to get datapoints belonging to cluster
        x = data_arr[cluster_assignments == cluster_number]

        # if there are no datapoints assignd to the cluster
        if x.shape[0] != 0:
            centroids_dict[cluster_number] = (np.sum(x, axis=0) / x.shape[0]).reshape(1, n_feat)
        else:
            centroids_dict[cluster_number] = np.sum(x, axis=0).reshape(1, n_feat)

    return centroids_dict


# initalizes cluster centroids using random parition method
def initialize_centroids(k):
    # dictionary to store centroids
    centroids_dict = {}

    # randomly assign datapoints to a cluster
    random_cluster_assignments = np.random.randint(low=0, high=k, size=(n_samp,), dtype=int)

    # return cluster centroids for random assignments
    return calculate_cluster_centroids(random_cluster_assignments)


# gets new cluster assignments given centroids using euclidean distance
def update_cluster_assignments(centroids_dict):
    # get number of clusters
    n_c = len(centroids_dict.keys())

    # array to store distance from each cluster
    dist_arr = np.zeros((n_samp, n_c), dtype=float)

    # caluclate euclidean distance from each centroid
    for cluster_number in range(n_c):
        dist_arr[:, cluster_number] = np.sqrt(np.sum((data_arr - centroids_dict[cluster_number]) ** 2, axis =1))

    # for each sample, index of the column of the lowest distance is closest cluster
    # and hence the cluster assignment
    cluster_assignments = np.argmin(dist_arr, axis = 1)

    return cluster_assignments


# calculates SSE error given final cluster assignments
def get_SSE(cluster_assignments):
    # compute centroids
    n_c = max(cluster_assignments) + 1

    # dictionary to store final centroids for computing SSE
    final_centroids_dict = calculate_cluster_centroids(cluster_assignments)

    SSE = 0

    for cluster_number in range(n_c):
        # use array slicing to get datapoints belonging to cluster
        x = data_arr[cluster_assignments == cluster_number]

        # compute euclidean distance
        dist_arr = np.sqrt(np.sum((x - final_centroids_dict[cluster_number]) ** 2, axis=1))

        # add sum of distances to sse
        SSE += np.sum(dist_arr)

    return SSE


# given an unlabeled data, returns k number of clusters
def get_kmeans_clusters(k):
    # initalize cluster centroids
    cluster_centroids_dict = initialize_centroids(k)

    # get new cluster assignments
    prev_cluster_assignments = update_cluster_assignments(cluster_centroids_dict)

    # i to count iterations
    i = 0
    SSE_final = 0  # final SSE
    SSE_20 = 0  # SSE after 20th iteration

    while True:

        # compute centroids
        cluster_centroids_dict = calculate_cluster_centroids(prev_cluster_assignments)

        if i == 19:
            SSE_20 = get_SSE(prev_cluster_assignments)

        # get new cluster assignments
        cluster_assignments = update_cluster_assignments(cluster_centroids_dict)

        # if new assignments is same as the old assignments, stop
        if (prev_cluster_assignments == cluster_assignments).all():
            break

        # save new assignments
        prev_cluster_assignments = cluster_assignments

        # increment i
        i += 1

    # get error
    SSE_final = get_SSE(cluster_assignments)

    # return final cluster assignments, sse after 20th iteration and final SSE
    return cluster_assignments, SSE_20, SSE_final


__, sse, __ = get_kmeans_clusters(5)

# initialize cluster numbers
k_values = list(np.arange(2, 20))

# list to store SSE values after 29th iteration
SSE_values = []

for k in k_values:
    __, sse, __ = get_kmeans_clusters(k)
    SSE_values.append(sse)
    print(f'SSE for {k} clusters is {sse:0.4f}.')
