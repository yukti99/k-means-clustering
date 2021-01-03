import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from Kmeans_clustering import KMeans


X, y = make_blobs(centers=4, n_samples=600, n_features=2, shuffle=True, random_state=42)
clusters_len = len(np.unique(y))
k = KMeans(K=clusters_len, max_iters=150, plot_steps=True)
y_pred = k.predict(X)
print(y_pred)
k.plot_data()



