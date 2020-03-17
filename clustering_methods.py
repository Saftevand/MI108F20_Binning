import numpy as np
from random import randint
import math
import pandas as pd

class clustering_element():
    def __init__(self, cluster, element):
        self.cluster = cluster
        self.element = element

class clustering_k_means():
    def __init__(self, k_clusters=10):
        self.k = k_clusters
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

    def do_clustering(self, dataset, max_iterations=10):
        prep_dataset = self.prep_data(dataset)
        self.forgy_init(dataset=dataset)
        self.assignment_step(dataset=prep_dataset)

        for i in range(max_iterations):
            self.update_step(dataset=prep_dataset)
            self.assignment_step(dataset=prep_dataset)

        clusters = []

        for i in range(0, self.k):
            current_cluster= []
            points_in_cluster = filter(lambda x: x.cluster == i, prep_dataset)
            for j in points_in_cluster:
                current_cluster.append(j.element)
            clusters.append(current_cluster)

        self.clustered_data = pd.DataFrame(clusters)