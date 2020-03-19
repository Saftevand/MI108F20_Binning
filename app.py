import autoencoders
import clustering_methods
import multiprocessing as _multiprocessing
import binner
import argparse
import data_processor


def main():

    args = handle_input_arguments()
    print(args)
    #   TODO    calc methyl
    fe = autoencoders.get_feature_extractor(args.featureextractor)()
    clustering_algorithm = clustering_methods.get_clustering(args.cluster)()

    _binner = binner.Binner(autoencoder=fe,
                            clustering_method=clustering_algorithm,
                            feature_matrix=data_processor.get_featurematrix())

    _binner.autoencoder.x_train, _binner.autoencoder.x_valid = \
        data_processor.get_train_and_validation_data(feature_matrix=_binner.feature_matrix, split_value=0.8)

    _binner.autoencoder.train()
    _binner.do_binning()




def handle_input_arguments():
    parser = argparse.ArgumentParser()

    #parser.add_argument("-r", "--read", help="Path to read", required=True)
    parser.add_argument("-r", "--read", help="Path to read")
    parser.add_argument("-b", "--bam", help="Path to BAM files")
    parser.add_argument("-c", "--cluster",nargs='?', default="KMeans", const="KMeans", help="Clustering algorithm to be used")
    parser.add_argument("-fe", "--featureextractor", nargs='?', default="DEC", const="DEC", help="Feature extractor to be used")
    return parser.parse_args()


if __name__ == '__main__':
    _multiprocessing.freeze_support()  # Skal være her så længe at vi bruger vambs metode til at finde depth
    main()

    '''
    binner_ = binner.Binner(autoencoder=autoencoders.DEC_greedy_autoencoder(train=None, valid=None), clustering_method= clustering_methods.clustering_k_means())

    binner_.extract_features()
    binner_.do_clustering()

    print("breakpoint")
    '''


    ''' Ting og sager til TSNE --> virker kun til linux (men virker det?)
    X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(encoded_train)

    plt.figure(figsize=(16, 10))
    sns.scatterplot(hue='y',
                    palette=sns.color_palette("hls", 10),
                    data=X_embedded,
                    legend="full",
                    alpha=0.3)
    '''