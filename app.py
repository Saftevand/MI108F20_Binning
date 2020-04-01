import clustering_methods
import multiprocessing as _multiprocessing
import binner
import argparse
import data_processor
import numpy as np
import tensorflow as tf
from tensorflow import keras
import clustering_methods


def main():
    '''Simon GPU fix'''
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass


    args = handle_input_arguments()
    print(args)
    #   TODO    calc methyl

    feature_matrix, contig_ids = data_processor.get_featurematrix(args.read, args.bam)

    print("Setup binner")
    binner_instance = binner.create_binner(split_value=1, clustering_method=args.clustering, binner_type=args.binnertype, feature_matrix=feature_matrix, contig_ids=contig_ids)

    print("Getting true bins")
    true_bins, contig_ids_true, contig_to_bin_id = data_processor.get_unique_ids_truth('D:/Downloads/out/gsa_mapping.binning')

    true_bins = data_processor.sort_bins_follow_input(contig_ids, contig_to_bin_id)

    print("Starting binning process")
    # Estimate ~ 60 bins in cami low
    # n_clusters er kun en ting for DEC

    # default adam params
    adam = keras.optimizers.Adam(learning_rate=0.001)

    binner_instance.do_binning(init='glorot_uniform',
                               pretrain_optimizer=adam,
                               n_clusters=60,
                               update_interval=140,
                               pretrain_epochs=100,
                               batch_size=128,
                               save_dir='results',
                               tolerance_threshold=1e-3,
                               max_iterations=100,
                               true_bins=true_bins)


    results = binner_instance.get_assignments()

    data_processor.write_bins_to_file(results)


def handle_input_arguments():
    parser = argparse.ArgumentParser()

    #parser.add_argument("-r", "--read", help="Path to read", required=True)
    parser.add_argument("-r", "--read", help="Path to read")
    parser.add_argument("-b", "--bam", help="Bam files", nargs='+')
    parser.add_argument("-c", "--clustering",nargs='?', default="KMeans", const="KMeans", help="Clustering algorithm to be used")
    parser.add_argument("-bt", "--binnertype", nargs='?', default="DEC", const="DEC", help="Binner type to be used")
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