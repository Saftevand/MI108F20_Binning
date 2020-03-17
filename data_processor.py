import vamb_tools
import numpy as np
from sklearn.preprocessing import normalize
import math

class Data_processor():
    def __init__(self):
        self.bins = None

    def get_tnfs(self, path=None):
        if path is None:
            path = 'test/bigfasta.fna.gz'
        with vamb_tools.Reader(path, 'rb') as filehandle:
            tnfs = vamb_tools.read_contigs(filehandle, minlength=4)
        return tnfs

    def get_depth(self, paths=None):
        if paths is None:
            paths = ['test/one.bam', 'test/two.bam',
                     'test/three.bam']
        return vamb_tools.read_bamfiles(paths)

    def get_featurematrix(self, use_depth=False, load_data=True):
        use_depth = use_depth
        load_data = load_data
        feature_matrix = None

        if (use_depth):
            if (load_data):
                tnfs = np.load('vamb_tnfs.npy')
                depth = np.load('vamb_depths.npy')
            else:
                tnfs = self.get_tnfs()
                depth = self.get_depth()
            norm_depth = normalize(depth, axis=1, norm='l1')
            feature_matrix = np.hstack([tnfs, norm_depth])
        else:
            if (load_data):
                feature_matrix = np.load('vamb_tnfs.npy')
            else:
                feature_matrix = self.get_tnfs()

        return feature_matrix


    def get_train_and_validation_data(self, feature_matrix, split_value=0.8):

        split_length = math.floor(len(feature_matrix) * split_value)
        train = feature_matrix[:split_length, :]
        validate = feature_matrix[split_length + 1:, :]

        return train, validate
