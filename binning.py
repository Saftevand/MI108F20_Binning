
class Binner:

    def __init__(self, dataset, feature_extractor, clustering):
        self.feature_matrix = dataset
        self.feature_extractor = feature_extractor
        self.clustering = clustering


    def bin(self):
        self.feature_extractor.extract_features()
        self.clustering.cluster()
