from abc import ABC, abstractmethod
import typing


class FeatureExtractor(ABC):


    @abstractmethod
    def extract_features(self):
        pass


class AutoEncoder(FeatureExtractor):

    def extract_features(self):
        pass


class DEC(AutoEncoder):


    def extract_features(self):
        pass


class VAE(AutoEncoder):


    def extract_features(self):
        pass


def get_feature_extractor(fe):
    return feature_extractor_dict[fe]


feature_extractor_dict = {
    'DEC': DEC,
    'VAE': VAE
}