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

    feature_matrix, contig_ids = data_processor.get_featurematrix(args)

    binner_instance = binner.create_binner(split_value=1, clustering_method=args.clustering,
                                           binner_type=args.binnertype, feature_matrix=feature_matrix,
                                           contig_ids=contig_ids)

    binner_instance.do_binning()

    results = binner_instance.get_assignments()

    data_processor.write_bins_to_file(results)


def simon_main(args):
    feature_matrix, contig_ids = data_processor.get_featurematrix(args.read, args.bam)
    print("Setup binner")
    binner_instance = binner.create_binner(split_value=1, clustering_method=args.clustering,
                                           binner_type=args.binnertype, feature_matrix=feature_matrix,
                                           contig_ids=contig_ids)

    print("Getting true bins")
    true_bins, contig_ids_true, contig_to_bin_id = data_processor.get_unique_ids_truth(
        'D:/Downloads/out/gsa_mapping.binning')

    true_bins = data_processor.sort_bins_follow_input(contig_ids, contig_to_bin_id)

    print("Starting binning process")
    # Estimate ~ 60 bins in cami low
    # n_clusters er kun en ting for DEC

    # rest are default adam params
    adam = keras.optimizers.Adam(learning_rate=0.1)

    binner_instance.do_binning(true_bins=true_bins,
                               init='glorot_uniform',
                               pretrain_optimizer='adam',
                               n_clusters=60,
                               update_interval=140,
                               pretrain_epochs=1000,
                               batch_size=128,
                               save_dir='results',
                               tolerance_threshold=1e-3,
                               max_iterations=1000)

    '''
    # glorot uniform is about the same as xavier.... keras stuff
    # pretrain optimizer is used in both layerwise training and finetuning
    binner_instance.do_binning(init='glorot_uniform',
                               pretrain_optimizer=adam,
                               neuron_list=[200, 200, 500, 10],
                               pretrain_epochs=10,
                               finetune_epochs=10,
                               pretrain_loss='mean_squared_error',
                               n_clusters=60,
                               update_interval=140,
                               batch_size=256,
                               tolerance_threshold=1e-3,
                               max_iterations=300,
                               save_dir='results',
                               true_bins=true_bins,
                               verbose=1)
    
    count = 1
    for i in binner_instance.layers_history:
        plt.plot(i.history['loss'])
        plt.title(f'layer loss. layer {count}')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc='upper left')
        plt.show()
        count += 1
    '''
    plt.plot(binner_instance.pretrain_hist.history['loss'])
    plt.title(f'Finetune loss.')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.show()

    plt.plot(binner_instance.cluster_loss_list, label='MAE (testing data)')
    plt.title('Clustering loss')
    plt.ylabel('Loss value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()

    results = binner_instance.get_assignments()

    data_processor.write_bins_to_file(results)

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

    parser.add_argument("-c", "--clustering",nargs='?', default="KMeans", const="KMeans", help="Clustering algorithm to be used")
    parser.add_argument("-bt", "--binnertype",nargs='?', default="DEC", const="DEC", help="Binner type to be used")
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
