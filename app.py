import autoencoders
import clustering_methods
import multiprocessing as _multiprocessing
import binner
import argparse
import data_processor
import tensorflow as tf
from tensorflow import keras

'''
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras
'''

def main():
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass


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
    data_processor.write_bins_to_file(_binner.bins)

    wait = input('Binning complete! Press any key to exit.')


def handle_input_arguments():
    parser = argparse.ArgumentParser()

    #parser.add_argument("-r", "--read", help="Path to read", required=True)
    parser.add_argument("-r", "--read", help="Path to read")
    parser.add_argument("-b", "--bam", help="Path to BAM files")
    parser.add_argument("-c", "--cluster",nargs='?', default="KMeans", const="KMeans", help="Clustering algorithm to be used")
    parser.add_argument("-fe", "--featureextractor", nargs='?', default="SAE", const="SAE", help="Feature extractor to be used")
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