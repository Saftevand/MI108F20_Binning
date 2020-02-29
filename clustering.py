from abc import ABC, abstractmethod



class Clustering(ABC):


    @abstractmethod
    def cluster(self):
        pass


class KMeans(Clustering):


    def cluster(self):
        pass


def get_clustering(cluster):

    return clustering_algorithms_dict[cluster]

clustering_algorithms_dict = {
    'KMeans': KMeans
}