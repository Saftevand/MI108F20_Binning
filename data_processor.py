import vamb_tools
import numpy as np
from sklearn.preprocessing import normalize
import math

def get_tnfs(path=None):
    if path is None:
        path = 'test/bigfasta.fna.gz'
    with vamb_tools.Reader(path, 'rb') as filehandle:
        tnfs, contig_names = vamb_tools.read_contigs(filehandle, minlength=4)

    return tnfs, contig_names


def get_depth(paths=None):
    if paths is None:
        paths = ['test/one.bam', 'test/two.bam',
                 'test/three.bam']
    return vamb_tools.read_bamfiles(paths)


def get_featurematrix(read_path, bam_path):
    load_data = False
    feature_matrix = None

    if (bam_path):
        if (load_data):
            tnfs = np.load('vamb_tnfs.npy')
            depth = np.load('vamb_depths.npy')
        else:
            tnfs, contig_ids = get_tnfs(read_path)
            depth = get_depth(bam_path)
        norm_depth = normalize(depth, axis=1, norm='l1')
        feature_matrix = np.hstack([tnfs, norm_depth])
    else:
        if (load_data):
            feature_matrix = np.load('vamb_tnfs.npy')
        else:
            feature_matrix, contig_ids = get_tnfs()

    return feature_matrix, contig_ids


def get_train_and_validation_data(feature_matrix, split_value=0.8):
    # TODO måske skal ham her være lidt bedre. Det er farligt at tage de første x % hver gang
    split_length = math.floor(len(feature_matrix) * split_value)
    train = feature_matrix[:split_length]
    validate = feature_matrix[split_length + 1:]
    return train, validate


def write_bins_to_file(bins):
    bins_string = "@@SEQUENCEID\tBINID\tLENGTH\n"

    for i in range(0, len(bins[0])):
        bins_string += f'{bins[0][i]}\t{bins[1][i]}\n'

    with open('binning_results.tsv', 'w') as output:
        output.write(bins_string)





