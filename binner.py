import clustering_methods
import autoencoders
import data_processor


class Binner:
    def __init__(self, autoencoder: autoencoders.Autoencoder, clustering_method: clustering_methods.clustering_method, extracted_features=None):
        self.autoencoder = autoencoder
        self.clustering_method = clustering_method
        self.feature_matrix = None
        self.extracted_features = extracted_features
        self.dataprocessor = data_processor.Data_processor()

    def extract_features(self):
        self.feature_matrix = self.dataprocessor.get_featurematrix()
        train, validation = self.dataprocessor.get_train_and_validation_data(feature_matrix=self.feature_matrix, split_value=0.8)

        self.autoencoder.x_train = train
        self.autoencoder.x_valid = validation
        self.autoencoder.train()
        self.extracted_features = self.autoencoder.extract_features(self.feature_matrix)

    def do_clustering(self):
        self.dataprocessor.bins = self.clustering_method.do_clustering(self.extracted_features)

    def do_binning(self):
        pass  # Tænker umiddelbart at det her er hvor bins bliver lavet til amber format?
