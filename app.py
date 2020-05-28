#import clustering_methods
import multiprocessing as _multiprocessing
#import binner
import argparse
import data_processor
import numpy as np
import tensorflow as tf
from tensorflow import keras
import clustering_methods
from tensorboard import program
from pathlib import Path
import newBinners
import os


def run_on_windows():

    dataset_path = 'D:\datasets\cami_high'
    tnfs, contig_ids, depth = data_processor.load_data_local(dataset_path)
    ids, contig_ids2, contigid_to_binid, contig_id_binid_sorted = data_processor.get_cami_data_truth(
        os.path.join(dataset_path, 'gsa_mapping_pool.binning'))
    labels = list(contig_id_binid_sorted.values())
    feature_matrix, x_train, x_valid, train_labels, validation_labels = data_processor.preprocess_data(tnfs=tnfs, depths=depth, labels=labels)

    binner_instance = newBinners.create_binner(binner_type='STACKED', feature_matrix=feature_matrix,
                                               contig_ids=contig_ids, labels=labels, x_train=x_train, x_valid=x_valid ,train_labels=train_labels, validation_labels=validation_labels)

    binner_instance.do_binning(load_model=False, load_clustering_AE=False)

    results = binner_instance.get_assignments()

    data_processor.write_bins_to_file(results)




def main():
    '''Simon GPU fix
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
    '''

    # DETTE SKULLE GERNE LÆSE DATA FRA CAMI BINS OG ARRANGE DATA ???????????
    print("Getting true bins")
    path = '/mnt/cami_high/gsa_mapping_pool.binning'
    ids, contig_ids2, contigid_to_binid, contig_id_binid_sorted = data_processor.get_cami_data_truth(path)

    labels = list(contig_id_binid_sorted.values())

    print("Starting binning process")

    args = handle_input_arguments()
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', args.outdir, '--host', '0.0.0.0', '--port', '13337'])
    url = tb.launch()

    feature_matrix, contig_ids, x_train, x_valid , train_labels, validation_labels = data_processor.get_featurematrix(args, labels)

    pretrain_params = {
        'learning_rate': 0.001,
        'reconst_loss': 'mae',
        'layer_size': 200,
        'num_hidden_layers': 3,
        'embedding_neurons': 32,
        'epochs': [100, 100, 150, 150],
        'batch_sizes': [32, 64, 128, 256],
        'activation_fn': 'elu',
        'regularizer': None,
        'initializer': 'he_normal',
        'denoise': False,
        'dropout': False,
        'drop_and_denoise_rate': 0.2,
        'BN': False,
        'sparseKLweight': 0.5,
        'sparseKLtarget': 0.1
    }
    clust_params = {
        'learning_rate': 0.0001,
        'loss_weights': [1, 0.05],  # [reconstruction, clustering]
        'batch_size': 4000,
        'epochs': 10,
        'reconst_loss': 'mse',
        'clust_loss': 'mse',
        'cuda': True,
        'eps': 0.5,
        'min_samples': 10
    }

    binner_instance = newBinners.create_binner(binner_type=args.binnertype, feature_matrix=feature_matrix,
                                               contig_ids=contig_ids, labels=labels, x_train=x_train, x_valid=x_valid,
                                               train_labels=train_labels, validation_labels=validation_labels,
                                               pretraining_params=pretrain_params, clust_params=clust_params)

    binner_instance.do_binning(load_model=True, load_clustering_AE=False)

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
    #run_on_windows()
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
