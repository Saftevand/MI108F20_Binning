import vamb_tools
import numpy as np
from sklearn.preprocessing import normalize
import math
import pandas as pd


def get_tnfs(path=None):
    if path is None:
        path = 'test/bigfasta.fna.gz'
    with vamb_tools.Reader(path, 'rb') as filehandle:
        tnfs, contig_names = vamb_tools.read_contigs(filehandle, minlength=4)

    dict= {}

    for i in range(0, len(tnfs)):
        dict[contig_names[i]] = tnfs[i]
    df = pd.DataFrame(data=dict).T

    return df


def get_depth(paths=None):
    if paths is None:
        paths = ['test/one.bam', 'test/two.bam',
                 'test/three.bam']
    return vamb_tools.read_bamfiles(paths)


def get_featurematrix(use_depth=False, load_data=False):
    use_depth = use_depth
    load_data = load_data
    feature_matrix = None

    if (use_depth):
        if (load_data):
            tnfs = np.load('vamb_tnfs.npy')
            depth = np.load('vamb_depths.npy')
        else:
            tnfs = get_tnfs()
            depth = get_depth()
        norm_depth = normalize(depth, axis=1, norm='l1')
        feature_matrix = np.hstack([tnfs, norm_depth])
    else:
        if (load_data):
            feature_matrix = np.load('vamb_tnfs.npy')
        else:
            feature_matrix = get_tnfs()

    return feature_matrix


def get_train_and_validation_data(feature_matrix, split_value=0.8):
    split_length = math.floor(len(feature_matrix) * split_value)
    train = feature_matrix[:split_length]
    validate = feature_matrix[split_length + 1:]
    return train, validate


def write_bins_to_file(bins):
    bins_string = "@@SEQUENCEID\tBINID\tLENGTH\n"

    for index in bins.index:
        bins_string += f'{index}\t{bins.loc[index]}\n'

    with open('binzz.tsv', 'w') as output:
        output.write(bins_string)





