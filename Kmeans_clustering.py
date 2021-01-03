"""
K MEANS CLUSTERING ALGORITHM
-> Clustering the data into k different clusters
-> using unsupervised learning as the dataset is unlabeled
-> Each sample is assigned to the cluster with the nearest mean

"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# so that we can reproduce the data later
np.random.seed(42)

# to find th Euclidean Distance between two vectors
# this will help in calculating the distance between each data point and cluster centers
def Euclidean_Dist(x1,x2):
    return np.sqrt( np.sum((x1-x2)**2))

class KMeans:
    num_plots=0
    # setting default values for the class
    # if k value not provided by user, k will be 5 and no of iterations will be 100
    def __init__(self, K=5, max_iters=100, plot_steps=False):

        # initializing values 
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # list of sample indices for each cluster
        # for each cluster we initialize an empty list
        self.clusters = [ [] for _ in range(self.K) ] 
        # mean feature vector for each cluster (actual samples)
        self.centroids = []

    # this method will input a list of chosen centroids 
    # and assign each data point/sample to its nearest centroid and return the clusters created
    def _create_clusters(self, centroids):
        # creating an empty list of K-lists for clusters
        clusters = [ [] for _ in range(self.K)]
        for index, sample in enumerate(self.X):
            # find the nearest centroid for current sample
            centroid_index  = self._nearest_centroid(sample,centroids)
            clusters[centroid_index].append(index)
        return clusters

    # returns the index of centroid in list which is nearest to the given sample
    def _nearest_centroid(self, sample, centroids):
        distances = [ Euclidean_Dist(sample,c) for c in centroids ]
        min_dist_index = np.argmin(distances)
        return min_dist_index

    def _get_centroids(self, clusters):
        # creating an array filled with zeroes with tuple of K and number of features
        centroids = np.zeros((self.K, self.n_features))
        # calculating the new centroid as the mean of all samples in the cluster
        # clusters is the list of lists
        for cluster_index, cluster in enumerate(clusters):
            # finding mean of samples in the current cluster
            cluster_mean = np.mean(self.X[cluster],axis=0)
            centroids[cluster_index] = cluster_mean
        return centroids

    def _is_converged(self, old_centroids, new_centroids):
        dist = [Euclidean_Dist(old_centroids[i], new_centroids[i]) for i in range(self.K)]
        # if there is no change in the distances of the old and new centroids, means the algorithm has converged
        if(sum(dist) == 0): 
            return True
        return False

    def _get_cluster_labels(self, clusters):        
        # creating an empty NumPy array for storing the label of each sample
        cluster_labels = [-1 for _ in range(self.n_samples)]
        for cluster_index, cluster in enumerate(clusters):
            for sample_index in cluster:
                cluster_labels[sample_index] = int(cluster_index) 
        for i in range(len(cluster_labels)):
            cluster_labels[i] = "Label-"+str(cluster_labels[i]) 
        return cluster_labels

    def predict(self, X):
        # for storing data
        self.X = X
        # number of samples and features
        self.n_samples, self.n_features = X.shape
        print("SAMPLES:", self.n_samples, self.n_features)

        # initialize the centroids
        # to randomly pick some samples
        # it will pick a random choice between 0 and number of samples
        # In KMeans algorithm initially, random samples are made centroids and gradually optimization is done and new centroids selected
        random_sample_indices = np.random.choice(self.n_samples, self.K, replace=False) # this will be an array of size self.K
        print("RANDOM SAMPLE INDICES = ",random_sample_indices)
        self.centroids = [self.X[i] for i in random_sample_indices]

        # optimization 
        for _ in range(self.max_iters):
            # update clusters
            self.clusters = self._create_clusters(self.centroids)
            if self.plot_steps:
                self.plot_data()

            # update centroids
            old_centroids = self.centroids
            self.centroids =  self._get_centroids(self.clusters)
            if self.plot_steps:
                self.plot_data()

            # checking for convergence of algorithm
            if self._is_converged(old_centroids, self.centroids):
                # we can end the clustering algorithm now
                break

        # return cluster_labels
        return self._get_cluster_labels(self.clusters)

    def plot_data(self):
        KMeans.num_plots+=1
        figure, x = plt.subplots(figsize=(12,8))
        for i,index in enumerate(self.clusters):
            # .T is for the transpose of the array
            point = self.X[index].T
            print("POINT = ",point)
            print()
            # to plot all the points in different colors for different clusters
            x.scatter(*point)
        # showing the centroids as markers to differentiate between them and normal data points 
        for point in self.centroids:
            x.scatter(*point, marker="x",color="black",linewidth=2)    
        plt.show()
        plt.title("K-Means Clustering Graph by Yukti Khurana: "+"Plot Number - "+str(KMeans.num_plots))
        plt.ylabel('Y-Position')
        plt.xlabel('X-Position')
        now = datetime.now()
        dt_string = now.strftime("%m%Y%H%M%S")
        name = str(dt_string)+" Plot Number - "+str(KMeans.num_plots)+".png"        
        #plt.show(block=False)
        plt.savefig("PlotImages/"+name)
        #plt.pause(3)
        plt.close()
        
        
        

    




