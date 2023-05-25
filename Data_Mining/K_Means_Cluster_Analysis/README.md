K Means Clustering is one of the simplest partitional clustering algorithm for unsupervised learning. Although the algorithm is straightforward, it requires the user to provide the number of clusters to create which does not always have a straighforward answer especially for unlabeled datasets under analysis. In addition, K Means Clustering also suffers from the "initial centriod problem" if centroids are assigned randomly. The initial centroids greatly determine whether the user might end up with a somewhat optimal or sub-optimal clustering result.

- To avoid the "initial centroid problem", a random partition method is used to initalize centroids where eahc point in the data is randomly assignned a cluster number and the aberage is taken at the end to yield inital points. This makes sure that the centroids start out in a region that lies at the center region of all datapoints. Although this is not the best solution and has it's own drawbacks it avoids the "initial centroid problem" in most scenarios. 

- As K Means requires the user to provide the value of 'k', this implementation runs the algorithm for different values of k and plots the SSE for each value, allowing the user to pick an appropriate value of k using the elbow method or some other strategy.

file(s):
1. KMeans.ipynb: 
- Jupyter notebook implementation for P3. 
- Automatically reads data and shows results for all 3 datasets in UCI_datasest folder. 
- At the end the user will be prompted to provide a filepath. 