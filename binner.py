import clustering_methods
import autoencoders
import data_processor


class Binner:
    def __init__(self, autoencoder: autoencoders.Autoencoder, clustering_method: clustering_methods.clustering_method, feature_matrix = None, extracted_features=None):
        self.autoencoder = autoencoder
        self.clustering_method = clustering_method
        self.feature_matrix = feature_matrix
        self.bins = None
        self.extracted_features = extracted_features

    def do_binning(self):
        self.extracted_features = self.autoencoder.extract_features(self.feature_matrix)
        self.bins = self.clustering_method.do_clustering(self.extracted_features)
