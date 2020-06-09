import multiprocessing as _multiprocessing
import argparse
import data_processor
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
        'callback_interval': 100,
        'abd_weight': 40
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
    'min_cluster_size': 6,
    'callback_interval': 10
}

def main():
    print("Starting binning process")
    args = handle_input_arguments()

    feature_matrix, contig_ids, x_train, x_valid , train_labels, validation_labels, samples = data_processor.get_featurematrix(args)
    pretrain_params['number_of_samples'] = samples

    binner_instance = newBinners.create_binner(binner_type=args.binnertype, feature_matrix=feature_matrix,
                                               contig_ids=contig_ids, x_train=x_train, x_valid=x_valid,
                                               train_labels=train_labels, validation_labels=validation_labels,
                                               pretraining_params=pretrain_params, clust_params=clust_params, debug=False)

    binner_instance.do_binning(load_model=False, load_clustering_AE=False)

    results = binner_instance.get_assignments(include_outliers=False)

    data_processor.write_bins_to_file(bins=results, output_dir=args.outdir)




def handle_input_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--fasta", help="Path to fasta file")
    parser.add_argument("-b", "--bam", help="Bam files", nargs='+')
    parser.add_argument("-o", "--outdir", help="Path to outdir of bins")
    parser.add_argument("-bt", "--binnertype", nargs='?', default="STACKED", const="STACKED",
                            help="Binner type to be used")

    parser.add_argument("-lt", "--loadtnfs", help="Path to tnfs.npy")
    parser.add_argument("-lc", "--loadcontigids", help="Path to contig_ids.npy")
    parser.add_argument("-ld", "--loaddepth", help="Path to depth.npy")


    args = parser.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    return args



if __name__ == '__main__':
    _multiprocessing.freeze_support()  # Skal være her så længe at vi bruger vambs metode til at finde depth
    main()
