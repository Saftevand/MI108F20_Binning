import numpy as np
from random import randint
import math
import pandas as pd
import abc
import cuml
import cudf
import sklearn


class clustering_method:
    def __init__(self):
        pass

    @abc.abstractmethod
    def do_clustering(self, dataset):
        pass

    @abc.abstractmethod
    def get_loss(self):
        pass
        # Assignments


class clustering_element():
    def __init__(self, cluster, element):
        self.cluster = cluster
        self.element = element


class random_cluster(clustering_method):
    def __init__(self, k_clusters=10):
        super().__init__()
        self.k = k_clusters
        self.clustered_data = None

    def do_clustering(self, dataset):
        clusters = [[] for i in range(self.k)]
        for contig in dataset:
            assignment = randint(0, self.k-1)
            list = clusters[assignment]
            list.append(contig)

        self.clustered_data = pd.DataFrame(clusters)
        return self.clustered_data


class clustering_k_means(clustering_method):
    def __init__(self, k_clusters=10, max_iterations=10):
        super().__init__()
        self.k = k_clusters
        self.max_ite = max_iterations

    def do_clustering(self, dataset):
        kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=self.k, max_iter=self.max_ite)
        kmeans.fit(dataset)
        return kmeans.labels_

    def get_loss(self):
        return None

'''
class clustering_k_means(clustering_method):
    def __init__(self, k_clusters=10, max_iterations=5):
        super().__init__()
        self.k = k_clusters
        self.max_iterations = max_iterations
        self.centroids = []
        self.clustered_data = None

    def forgy_init(self, dataset):
        length = len(dataset)
        blacklist = np.zeros(self.k)
        rand = randint(0, length-1)
        for i in range(self.k):
            while rand in blacklist:
                rand = randint(0, length-1)
            self.centroids.append(dataset[rand])
            blacklist[i] = rand

    def distance(self, a, b):  # <-- Euclidean distance
        sum_ab = 0
        for i in a:
            for j in b:
                sum_ab += i**2 - j**2
        sum_ab = math.sqrt(abs(sum_ab))
        return sum_ab

    def assignment_step(self, dataset):
        for element in dataset:
            min_distance = float("inf")

            for i in range(0, len(self.centroids)):
                dist = self.distance(element.element, self.centroids[i])
                if(dist < min_distance):
                    min_distance = dist
                    element.cluster = i

    def update_step(self, dataset):
        for i in range(0, len(self.centroids)):
            points_in_cluster = filter(lambda x: x.cluster == i, dataset)
            points_in_cluster2 = []
            is_init = True
            new_centroid= []
            for x in points_in_cluster:
                points_in_cluster2.append(x)
                if (is_init):
                    new_centroid = x.element
                else:
                    for i in range(0, len(new_centroid)):
                        new_centroid[i] += x.cluster[i]

            length = len(points_in_cluster2)
            for k in new_centroid:
                k = k / length

            self.centroids[i] = new_centroid

    def prep_data(self, dataset):
        result = []
        for i in range(0, len(dataset)):
            result.append(clustering_element(element=dataset[i], cluster=-1))
        return result

    def do_clustering(self, dataset):
        prep_dataset = self.prep_data(dataset)
        self.forgy_init(dataset=dataset)
        self.assignment_step(dataset=prep_dataset)

        for i in range(self.max_iterations):
            self.update_step(dataset=prep_dataset)
            self.assignment_step(dataset=prep_dataset)

        clusters = []

        for i in range(0, self.k):
            current_cluster = {}
            points_in_cluster = filter(lambda x: x.cluster == i, prep_dataset)
            for j in points_in_cluster:
                current_cluster[j.name] = j.cluster
            clusters.append(current_cluster)

        result = pd.DataFrame(clusters)
        result = result.sum(axis=0, skipna=True)

        self.clustered_data = result
        return self.clustered_data
'''

class DBSCAN_GPU(clustering_method):
    def __init__(self, eps=1.0, min_samples=5): #eps skal være en float værdi
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples

    def do_clustering(self, dataset, max_iterations=5):
        dbscan = cuml.cluster.DBSCAN(eps=self.eps, min_samples=self.min_samples)
        dbscan.fit(dataset)
        return dbscan.labels_.to_array()

    def get_loss(self):
        return None


class KMEANS_GPU(clustering_method):
    def __init__(self, k_amount_of_clusters=5, max_iterations=5):
        super().__init__()
        self.k = k_amount_of_clusters
        self.max_iterations = max_iterations

    def do_clustering(self, dataset):
        kmeans = cuml.cluster.KMeans(n_clusters=self.k, max_iter=self.max_iterations)
        kmeans.fit(dataset)
        return kmeans.labels_.to_array()

    def get_loss(self):
        return None

