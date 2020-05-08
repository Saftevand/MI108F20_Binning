import clustering_methods
import multiprocessing as _multiprocessing
import binner
import argparse
import data_processor
import numpy as np
import tensorflow as tf
from tensorflow import keras
import clustering_methods
from pathlib import Path


def main():
    '''Simon GPU fix
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
    '''

    ''' DETTE SKULLE GERNE LÆSE DATA FRA CAMI BINS OG ARRANGE DATA ???????????
    print("Getting true bins")
    true_bins, contig_ids_true, contig_to_bin_id = data_processor.get_unique_ids_truth(
        'D:/Downloads/out/gsa_mapping.binning')

    true_bins = data_processor.sort_bins_follow_input(contig_ids, contig_to_bin_id)

    print("Starting binning process")
    '''

    args = handle_input_arguments()
    print(args)
    #   TODO    calc methyl

    #binid_to_int = data_processor.get_unique_ids_truth('/mnt/cami_high/gsa_mapping_pool.binning')

    #print(len(binid_to_int.keys()))


    feature_matrix, contig_ids = data_processor.get_featurematrix(args)

    binner_instance = binner.create_binner(split_value=0.8, clustering_method=args.clustering,
                                           binner_type=args.binnertype, feature_matrix=feature_matrix,
                                           contig_ids=contig_ids, log_dir=args.outdir)



    binner_instance.do_iris_binning()

    results = binner_instance.get_assignments()

    data_processor.write_bins_to_file(results)

    print(feature_matrix.mean())
    print(f'{feature_matrix.max()} : max')
    print(f'{feature_matrix.min()} : min')


def handle_input_arguments():
    parser = argparse.ArgumentParser()

    group1 = parser.add_mutually_exclusive_group(required=True)

    group2 = parser.add_mutually_exclusive_group(required=True)

    #parser.add_argument("-r", "--read", help="Path to read", required=True)
    group1.add_argument("-r", "--read", help="Path to read")
    group1.add_argument("-lt", "--loadtnfs", help="Path to tnfs.npy")
    parser.add_argument("-lc", "--loadcontigids", help="Path to contig_ids.npy")

    group2.add_argument("-b", "--bam", help="Bam files", nargs='+')
    group2.add_argument("-ld", "--loaddepth", help="Path to depth.npy")

    parser.add_argument("-st", "--savepathtnfs", help="Path to save tnfs")
    parser.add_argument("-sc", "--savepathcontigids", help="Path to save contigids")

    parser.add_argument("-c", "--clustering",nargs='?', default="KMeans_gpu", const="KMeans_gpu", help="Clustering algorithm to be used")
    parser.add_argument("-bt", "--binnertype",nargs='?', default="SEQ", const="SEQ", help="Binner type to be used")
    parser.add_argument("-sd", "--savepathdepth", help="Path to save depths")

    parser.add_argument("-o", "--outdir", required=True,  help="Path to outdir of bins")

    args = parser.parse_args()

    if args.savepathtnfs and not args.read:
        parser.error("-st requires -r")
    if args.savepathcontigids and not args.read:
        parser.error("-sc requires -r")
    if args.savepathdepth and not args.bam:
        parser.error("-sd requires -b")

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    return args


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
