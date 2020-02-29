import binning
import feature_extractor
import clustering
import pandas as pd
import numpy as np
import sys
import os
import typing
import argparse






def main():

    args = handle_input_arguments()
    print(args)
    #   TODO    calc input_features, calc tnf, abundance, methyl
    #   TODO    Make dataframe of features
    feature_matrix = pd.DataFrame(np.arange(10))
    fe = feature_extractor.get_feature_extractor(args.featureextractor)()
    clustering_algorithm = clustering.get_clustering(args.cluster)()

    binner = binning.Binner(feature_matrix,feature_extractor=fe, clustering=clustering_algorithm)
    test = 10


def handle_input_arguments():
    parser = argparse.ArgumentParser()

    #parser.add_argument("-r", "--read", help="Path to read", required=True)
    parser.add_argument("-r", "--read", help="Path to read")
    parser.add_argument("-b", "--bam", help="Path to BAM files")
    parser.add_argument("-c", "--cluster",nargs='?', default="KMeans", const="KMeans", help="Clustering algorithm to be used")
    parser.add_argument("-fe", "--featureextractor", nargs='?', default="VAE", const="VAE", help="Feature extractor to be used")
    return parser.parse_args()


if __name__ == '__main__':
    main()


#   Required arguments:
#       Path to reads/contigs
#       Path to BAM files

#   Optional arguments
#       Feature extractor
#       Clustering method

