#import clustering_methods
import multiprocessing as _multiprocessing
#import binner
import argparse
import data_processor
import hdbscan
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
#import clustering_methods
import sys
import json
from tensorboard import program
from pathlib import Path
import newBinners
import os
import matplotlib
matplotlib.use('Agg')

pretrain_params = {
        'learning_rate': 0.001,
        'reconst_loss': 'mae',
        'layer_size': 200,
        'num_hidden_layers': 4,
        'embedding_neurons': 32,
        'epochs': [1],
        'batch_sizes': [256],
        #'epochs': [5, 5, 5],
        #'batch_sizes': [1024, 2048, 4096],
        'activation_fn': 'elu',
        'regularizer': None,
        'initializer': 'he_normal',
        'optimizer': 'Adam',
        'denoise': False,
        'dropout': False,
        'drop_and_denoise_rate': 0.2,
        'BN': False,
        'sparseKLweight': 0.1,
        'sparseKLtarget': 0.05,
        'jacobian_weight': 1e-3,
        'callback_interval': 1,
        'abd_weight': 2
    }
clust_params = {
    'learning_rate': 0.001,
    'optimizer': 'Adam',
    'loss_weights': [1, 0.05],  # [reconstruction, clustering]
    'gaussian_bandwidth': 1,
    'jacobian_weight': 1e-4,
    'clustering_weight': 0.05,
    'epochs': 30,
    'reconst_loss': 'mae',
    'clust_loss': 'mae', #virker ikke for stacked --> default = gaussianloss
    'cuda': True,
    'eps': 0.5,
    'min_samples': 2,
    'min_cluster_size': 10,
    'callback_interval': 10
}

def run_on_windows(config, pretraining_params, clust_param):


    pretraining_params = pretraining_params
    clustering_params = clust_param

    if config:
        pretraining_params, clustering_params = load_training_config(config)

    dataset_path = 'D:\\datasets\\cami_airways'
    #dataset_path = 'D:/datasets/cami_medium'
    tnfs, contig_ids, depth = data_processor.load_data_local(dataset_path)
    pretraining_params['number_of_samples'] = depth.shape[1]
    print(pretraining_params['number_of_samples'])
    #ids, contig_ids2, contigid_to_binid, contig_id_binid_sorted = data_processor.get_cami_data_truth(
    #    os.path.join(dataset_path, 'gsa_mapping_pool.binning'))
    #labels = list(contig_id_binid_sorted.values())
    labels = []
    feature_matrix, x_train, x_valid, train_labels, validation_labels = data_processor.preprocess_data(tnfs=tnfs, depths=depth, labels=labels, use_validation_data=False)

    binner_instance = newBinners.create_binner(binner_type='SPARSE', feature_matrix=feature_matrix,
                                               contig_ids=contig_ids, labels=labels, x_train=x_train, x_valid=x_valid ,train_labels=train_labels, validation_labels=validation_labels, clust_params=clustering_params, pretraining_params=pretraining_params)


    binner_instance.do_binning(load_model=False, load_clustering_AE=False)

    #bins = binner_instance.get_assignments(include_outliers=False)
    #data_processor.write_bins_to_file(bins)
    #hdbscan_non_embedded_data(binner=binner_instance)
    run_amber(binner_instance.log_dir, cami_high=True)

    #run_amber('/home/lasse/MI108F20_Binning/Logs/run_2020_06_06-14_29_29_STACKED/')

    #run_amber('/home/lasse/MI108F20_Binning/HDBSCAN_results/cami_medium/', cami_medium=True)

    #run_amber('/home/lasse/MI108F20_Binning/Logs/STACKED_4x200-32_Drp_0_Lr_0.001_run_08-14_03_17/', cami_airways=True)
def load_training_config(config_path):
    with open(config_path, 'r') as config:
        params = json.load(config)
        return params['pretraining_params'], params['clust_params']


def run_amber(path, cami_high=False, cami_medium=False, cami_airways=False):

    directory_of_files = os.path.join(os.path.abspath(path), "*binning_results.tsv")
    labels = glob.glob(directory_of_files)
    print(labels)
    labels = [label.split('/')[-1].split('binning')[0] for label in labels]
    paths_to_results = [os.path.abspath(x) for x in glob.glob(directory_of_files)]
    if cami_high:
        gold_standard_file = os.path.abspath('ground_truth_with_length_cami_high.tsv')
        unique_common_file = os.path.abspath('unique_common_cami_high.tsv')
    elif cami_medium:
        gold_standard_file = os.path.abspath('ground_truth_with_length_cami_medium.tsv')
        unique_common_file = os.path.abspath('unique_common_cami_medium.tsv')
    elif cami_airways:
        gold_standard_file = os.path.abspath('cami_airways_ground_truth_with_length.tsv')
        outdir_with_circular = os.path.join(path, 'amber_with_circular')
        command_amber_with_circular = f'amber.py -g {gold_standard_file} -l "{", ".join(labels)}" {" ".join(paths_to_results)} -o {outdir_with_circular}'
        os.system(command_amber_with_circular)
        return

    else:
        return
    outdir_with_circular = os.path.join(path, 'amber_with_circular')
    outdir_without_circular = os.path.join(path, 'amber_without_circular')
    command_amber_without_circular = f'amber.py -g {gold_standard_file} -l "{", ".join(labels)}" -r {unique_common_file} -k "circular element" {" ".join(paths_to_results)} -o {outdir_without_circular}'
    command_amber_with_circular = f'amber.py -g {gold_standard_file} -l "{", ".join(labels)}" {" ".join(paths_to_results)} -o {outdir_with_circular}'
    os.system(command_amber_with_circular)
    os.system(command_amber_without_circular)


def hdbscan_non_embedded_data(binner):

    hdbscan_instance = hdbscan.HDBSCAN(min_cluster_size=binner.clust_params['min_cluster_size'],
                                       min_samples=binner.clust_params['min_samples'], core_dist_n_jobs=36)
    all_assignments = hdbscan_instance.fit_predict(binner.x_train)
    binner.bins = all_assignments
    bins_without_outliers = binner.get_assignments(include_outliers=False)
    data_processor.write_bins_to_file(bins_without_outliers,
                                      output_dir='HDBSCAN_normalized_original')
    # self.binner.autoencoder.save(os.path.join(self.binner.log_dir, f'{self.prefix}Epoch_{epoch}'))


    # RUN unnormalized

    hdbscan_instance1 = hdbscan.HDBSCAN(min_cluster_size=binner.clust_params['min_cluster_size'],
                                        min_samples=binner.clust_params['min_samples'], core_dist_n_jobs=36)
    all_assignments = hdbscan_instance1.fit_predict(binner.feature_matrix)
    binner.bins = all_assignments
    bins_without_outliers = binner.get_assignments(include_outliers=False)
    data_processor.write_bins_to_file(bins_without_outliers, output_dir='HDBSCAN_unnormalized_original')



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



    binner_instance = newBinners.create_binner(binner_type=args.binnertype, feature_matrix=feature_matrix,
                                               contig_ids=contig_ids, labels=labels, x_train=x_train, x_valid=x_valid,
                                               train_labels=train_labels, validation_labels=validation_labels,
                                               pretraining_params=pretrain_params, clust_params=clust_params)

    binner_instance.do_binning(load_model=True, load_clustering_AE=True)

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

    parser.add_argument("-bt", "--binnertype",nargs='?', default="STACKED", const="STACKED", help="Binner type to be used")
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

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to config file")
    args = parser.parse_args()
    run_on_windows(args.config, pretrain_params, clust_params)
    #main()

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