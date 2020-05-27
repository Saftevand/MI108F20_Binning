import numpy as np
import random
import tensorflow as tf
from tensorboard.plugins import projector
#import matplotlib.pyplot as plt
from tensorboard import program
import time
from sklearn import decomposition
import seaborn as sns
import matplotlib.pylab as plt
import os
from collections import Counter
from collections import defaultdict
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
dataset_path = "/mnt/cami_high"
log_dir = os.path.join('Logs', time.strftime("run_%Y_%m_%d-%H_%M_%S"))
#log_dir = os.path.join('Logs', "PCA50")



tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', 'Logs', '--host', '0.0.0.0', '--port', '13337'])
url = tb.launch()

def load_data():
    tnfs = np.load(os.path.join(dataset_path, "tnfs_high.npy"))
    contig_ids = np.load(os.path.join(dataset_path, "contig_ids_high.npy"))
    depth = np.load(os.path.join(dataset_path, "depths_high.npy"))

    return tnfs, contig_ids, depth


def preprocess_data(tnfs, depths):

    tnf_shape = tnfs.shape[1]
    depth_shape = depths.shape[1]
    number_of_features = tnf_shape + depth_shape
    tnf_weight = tnf_shape / number_of_features
    depth_weight = depth_shape / number_of_features
    weighted_tnfs = tnfs * tnf_weight
    weighted_depths = depths * depth_weight

    #feature_matrix = np.hstack([weighted_tnfs, weighted_depths])
    feature_matrix = np.hstack([tnfs, depths])
    x_train, x_valid = train_test_split(feature_matrix, test_size=0.2, shuffle=True)
    training_mean = np.mean(x_train, axis=0)
    training_std = np.std(feature_matrix, axis=0)

    x_train -= training_mean
    x_train /= training_std

    x_valid -= training_mean
    x_valid /= training_std
    feature_matrix -= training_mean
    feature_matrix /= training_std
    return feature_matrix, x_train, x_valid


def get_cami_data_truth(path):
    with open(path, 'r') as input_file:
        # skips headers
        for _ in range(0, 4):
            next(input_file)

        ids = []
        contig_ids = []
        binid_to_int = defaultdict()
        contigid_to_binid = defaultdict()
        bin_id_to_contig_names = defaultdict(list)

        new_id = 0
        for line in input_file:
            line_elems = line.split('\t')
            bin_id = line_elems[1]
            cont_name = line_elems[0]
            contig_ids.append(line_elems[0])

            if bin_id in binid_to_int:
                id_int = binid_to_int[bin_id]

                # used for mapping contig id to bin id
                contigid_to_binid[cont_name] = id_int
            else:
                # new bin id
                binid_to_int[bin_id] = new_id
                id_int = new_id

                # used for mapping contig id to bin id
                contigid_to_binid[cont_name] = id_int

                new_id += 1

            ids.append(id_int)

    for contig_name, binid in contigid_to_binid.items():
        bin_id_to_contig_names[binid].append(contig_name)
    d = {int(k.split("|C")[1]): int(v) for k, v in contigid_to_binid.items()}
    contig_id_binid_sorted = dict(sorted(d.items()))
    return ids, contig_ids, contigid_to_binid, contig_id_binid_sorted


def project_data(data, contig_names, contig_id_to_bin_id, bins_to_plot=10, run_on_all_data=False, number_components=10):

    number_of_contigs = data.shape[0]
    random.seed(2)
    counter = Counter()
    for bin_id in contig_id_to_bin_id.values():
        counter[bin_id] += 1
    #bins_atleast_10_contigs = [key for key, val in counter.items() if val >= 10]

    sorted_by_count = sorted(list(counter.items()), key=lambda x: x[1], reverse=True)
    samples_of_bins = [fst for fst, snd in sorted_by_count[1:11]]
    #samples_of_bins = random.choices(bins_atleast_10_contigs, k=bins_to_plot)

    if run_on_all_data:
        pca = decomposition.PCA(n_components=number_components)
        pca.fit(data)
        data_to_project = pca.transform(data)
        plt.scatter(data_to_project[:, 0], data_to_project[:, 1])

        with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
            for _, bin_id in contig_id_to_bin_id.items():
                if bin_id in samples_of_bins:
                    f.write(f'{bin_id}\n')
                else:
                    f.write(f'-1\n')
    else:
        assignments = []
        mask = np.zeros(number_of_contigs, dtype=bool)
        #with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
        for contig_idx, bin_id in contig_id_to_bin_id.items():
            if bin_id in samples_of_bins:
                mask[contig_idx] = True
                assignments.append(bin_id)
                #f.write(f'{bin_id}\t{} \t\n')

        data_to_project = data[mask]
        pca = decomposition.PCA(n_components=number_components)
        pca.fit(data_to_project)
        data_to_project = pca.transform(data_to_project)

        with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
            f.write(f'Label\tpc1\tpc2\n')
            for idx, bin_assignment in enumerate(assignments):
                f.write(f'{bin_assignment}\t{data_to_project[idx][0]}\t{data_to_project[idx][1]}\n')
        sns.scatterplot(data_to_project[:, 0], data_to_project[:, 1])
        plt.show()

    #variance_of_components = pca.explained_variance_ratio_

    data_variable = tf.Variable(data_to_project)
    checkpoint = tf.train.Checkpoint(embedding=data_variable)
    checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

    # Set up config
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
    # config.model_checkpoint_path = os.path.join("embedding.ckpt")
    embedding.tensor_name = f'embedding/.ATTRIBUTES/VARIABLE_VALUE'
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(log_dir, config)
    #print(f'Variance explained by first two components: {variance_of_components}')


def stacked_autoencoder(x_train, x_valid, no_layers=3, no_neurons_hidden=200, no_neurons_embedding=32, epochs=100, drop=False, bn=False):
    #for i, val in enumerate(x_train):
    #    print(f'Avg {i}: \t {np.mean(val)}')
    #split data
    input_shape = x_train.shape[1]
    activation_fn = 'elu'
    init_fn = 'he_normal'
    #regularizer = tf.keras.regularizers.l2()
    regularizer = None

    #Create input layer
    encoder_input = tf.keras.layers.Input(shape=(input_shape,), name="input")
    layer = encoder_input

    if no_layers == 0:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        init_fn = tf.keras.initializers.RandomNormal()

        layer = tf.keras.layers.Dense(units=no_neurons_embedding, activation=activation_fn, kernel_initializer=init_fn, kernel_regularizer=regularizer, name=f'Encoder')(layer)
        out = tf.keras.layers.Dense(units=input_shape, kernel_initializer=init_fn, name='Decoder_output_layer')(layer)

        small_AE = tf.keras.Model(encoder_input, out, name='Decoder')
        small_AE.compile(optimizer=optimizer, loss='mae')
        small_AE.fit(x_train, x_train, epochs=epochs, validation_data=(x_valid, x_valid), shuffle=True)
        return small_AE


    #Create encoding layers
    for layer_index in range(no_layers):
        layer = tf.keras.layers.Dropout(.2)(layer) if drop else layer
        layer = tf.keras.layers.BatchNormalization()(layer) if bn else layer
        layer = tf.keras.layers.Dense(units=no_neurons_hidden, activation=activation_fn, kernel_initializer=init_fn, kernel_regularizer=regularizer,
                                      name=f'encoding_layer_{layer_index}')(layer)


    #create embedding layer
    layer = tf.keras.layers.BatchNormalization()(layer) if bn else layer
    embedding_layer = tf.keras.layers.Dense(units=no_neurons_embedding, kernel_initializer=init_fn,
                                  name='embedding_layer')(layer)


    # Create decoding layers

    decoder_input = tf.keras.Input(shape=(no_neurons_embedding,), name='embedding')
    layer = decoder_input
    for layer_index in range(no_layers):
            layer = tf.keras.layers.Dropout(.2)(decoder_input) if drop else layer
            layer = tf.keras.layers.BatchNormalization()(layer) if bn else layer
            layer = tf.keras.layers.Dense(units=no_neurons_hidden, activation=activation_fn, kernel_initializer=init_fn, kernel_regularizer=regularizer,
                                          name=f'decoding_layer_{layer_index}')(layer)
    layer = tf.keras.layers.Dropout(.2)(layer) if drop else layer
    layer = tf.keras.layers.BatchNormalization()(layer) if bn else layer
    decoder_output = tf.keras.layers.Dense(units=input_shape, kernel_initializer=init_fn,
                                  name='Decoder_output_layer')(layer)

    encoder = tf.keras.Model(encoder_input, embedding_layer, name='Encoder')
    decoder = tf.keras.Model(decoder_input,decoder_output, name='Decoder')
    stacked_ae_input = tf.keras.layers.Input(shape=(input_shape,), name="Input_features")
    embedded_features = encoder(stacked_ae_input)
    reconstructed_features = decoder(embedded_features)

    stacked_ae = tf.keras.Model(stacked_ae_input, reconstructed_features, name='autoencoder')

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    var = stacked_ae.get_layer('Encoder').get_layer('embedding_layer')
    def create_loss(layer):
        def contractive_loss(y_pred, y_true):
            mse = K.mean(K.square(y_true - y_pred), axis=1)

            W = K.variable(value=layer.get_weights()[0])  # N x N_hidden
            W = K.transpose(W)  # N_hidden x N
            h = layer.output
            dh = h * (1 - h)  # N_batch x N_hidden
            lam = 1e-4
            # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
            contractive = lam * K.sum(dh ** 2 * K.sum(W ** 2, axis=1), axis=1)

            return mse + contractive
        return contractive_loss


    stacked_ae.compile(optimizer=optimizer, loss='mae')

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    print(encoder.summary())
    print(decoder.summary())
    print(stacked_ae.summary())


    history = stacked_ae.fit(x_train, x_train, epochs=50, batch_size=32, validation_data=(x_valid, x_valid),
                             shuffle=True, callbacks=[tensorboard_callback])
    history = stacked_ae.fit(x_train, x_train, epochs=50, batch_size=64, validation_data=(x_valid, x_valid),
                             shuffle=True, callbacks=[tensorboard_callback])
    history = stacked_ae.fit(x_train, x_train, epochs=100, batch_size=128, validation_data=(x_valid, x_valid),
                             shuffle=True, callbacks=[tensorboard_callback])
    history = stacked_ae.fit(x_train, x_train, epochs=150, batch_size=256, validation_data=(x_valid, x_valid),
                             shuffle=True, callbacks=[tensorboard_callback])
    history = stacked_ae.fit(x_train, x_train, epochs=200, batch_size=512, validation_data=(x_valid, x_valid),
                             shuffle=True, callbacks=[tensorboard_callback])
    history = stacked_ae.fit(x_train, x_train, epochs=200, batch_size=1024, validation_data=(x_valid, x_valid),
                             shuffle=True, callbacks=[tensorboard_callback])

    return stacked_ae

def hvordan_er_det_her_ikke_måden_at_lave_en_AE_på(x_train, x_valid, no_layers=3, no_neurons_hidden=200, no_neurons_embedding=32, epochs=100, drop=False, bn=False, denoise=False):
    if drop and denoise:
        print("HOLD STOP. DROPOUT + DENOISING det er for meget")
        return

    #split data
    input_shape = x_train.shape[1]
    activation_fn = 'elu'
    init_fn = 'he_normal'
    #regularizer = tf.keras.regularizers.l2()
    regularizer = None

    #Create input layer
    encoder_input = tf.keras.layers.Input(shape=(input_shape,), name="input")
    layer = encoder_input

    if no_layers == 0:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        init_fn = tf.keras.initializers.RandomNormal()

        layer = tf.keras.layers.Dense(units=no_neurons_embedding, activation=activation_fn, kernel_initializer=init_fn, kernel_regularizer=regularizer, name=f'Encoder')(layer)
        out = tf.keras.layers.Dense(units=input_shape, kernel_initializer=init_fn, name='Decoder_output_layer')(layer)

        small_AE = tf.keras.Model(encoder_input, out, name='Decoder')
        small_AE.compile(optimizer=optimizer, loss='mae')
        small_AE.fit(x_train, x_train, epochs=epochs, validation_data=(x_valid, x_valid), shuffle=True)
        return small_AE

    layer = tf.keras.layers.Dropout(.2)(layer) if denoise else layer

    #Create encoding layers
    for layer_index in range(no_layers):
        layer = tf.keras.layers.Dropout(.2)(layer) if drop else layer
        layer = tf.keras.layers.BatchNormalization()(layer) if bn else layer
        layer = tf.keras.layers.Dense(units=no_neurons_hidden, activation=activation_fn, kernel_initializer=init_fn, kernel_regularizer=regularizer,
                                      name=f'encoding_layer_{layer_index}')(layer)

    #create embedding layer
    layer = tf.keras.layers.BatchNormalization()(layer) if bn else layer
    embedding_layer = tf.keras.layers.Dense(units=no_neurons_embedding, kernel_initializer=init_fn,
                                  name='latent_layer')(layer)


    # Create decoding layers
    layer = embedding_layer

    for layer_index in range(no_layers):
            layer = tf.keras.layers.Dropout(.2)(layer) if drop else layer
            layer = tf.keras.layers.BatchNormalization()(layer) if bn else layer
            layer = tf.keras.layers.Dense(units=no_neurons_hidden, activation=activation_fn, kernel_initializer=init_fn, kernel_regularizer=regularizer,
                                          name=f'decoding_layer_{layer_index}')(layer)

    layer = tf.keras.layers.Dropout(.2)(layer) if drop else layer
    layer = tf.keras.layers.BatchNormalization()(layer) if bn else layer
    decoder_output = tf.keras.layers.Dense(units=input_shape, kernel_initializer=init_fn,
                                  name='Decoder_output_layer')(layer)


    stacked_ae = tf.keras.Model(encoder_input, decoder_output, name='autoencoder')

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    stacked_ae.compile(optimizer=optimizer, loss='mae')

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    print(stacked_ae.summary())


    history = stacked_ae.fit(x_train, x_train, epochs=50, batch_size=32, validation_data=(x_valid, x_valid),
                             shuffle=True, callbacks=[tensorboard_callback])
    history = stacked_ae.fit(x_train, x_train, epochs=50, batch_size=64, validation_data=(x_valid, x_valid),
                             shuffle=True, callbacks=[tensorboard_callback])
    history = stacked_ae.fit(x_train, x_train, epochs=100, batch_size=128, validation_data=(x_valid, x_valid),
                             shuffle=True, callbacks=[tensorboard_callback])
    history = stacked_ae.fit(x_train, x_train, epochs=150, batch_size=256, validation_data=(x_valid, x_valid),
                             shuffle=True, callbacks=[tensorboard_callback])
    history = stacked_ae.fit(x_train, x_train, epochs=200, batch_size=512, validation_data=(x_valid, x_valid),
                             shuffle=True, callbacks=[tensorboard_callback])
    history = stacked_ae.fit(x_train, x_train, epochs=200, batch_size=1024, validation_data=(x_valid, x_valid),
                             shuffle=True, callbacks=[tensorboard_callback])

    return stacked_ae

def sensitivity(autoencoder, dataset):

    dataset_variable = tf.Variable(dataset)
    with tf.GradientTape() as tape:
        reconstructed = autoencoder(dataset_variable, training=True)
        loss = tf.reduce_mean(tf.keras.losses.MAE(dataset_variable, reconstructed))

    gradients = tape.gradient(loss, dataset_variable)
    #gradients = tf.reduce_mean(tf.abs(gradients), 0)
    gradients = tf.reduce_mean(gradients, 0)
    number_of_features = dataset.shape[1]
    tnf_feature = [f'TNF\n{e}' for e in range(1,137)]
    abundance_feature = [f'ABD\n{e}' for e in range(1,number_of_features-135)]
    shape = int(np.ceil(np.sqrt(number_of_features)))
    labels = tnf_feature + abundance_feature
    labels += list(range(0, (shape*shape)-number_of_features))
    mask = np.zeros(shape*shape,dtype=bool)
    for index in range(number_of_features, shape*shape):
        mask[index] = True
    mask = mask.reshape(shape,shape)
    labels = np.asarray(labels).reshape(shape, shape)
    gradient_list = np.asarray(gradients.numpy()).tolist()
    max_gradient = max(gradient_list)
    min_gradient = min(gradient_list)
    for _ in range((shape * shape) - number_of_features):
        gradient_list.append(0.0)
    np_gradients = np.asarray(gradient_list).reshape(shape,shape)
    fig, ax = plt.subplots(figsize=(20,20))
    ax = sns.heatmap(np_gradients, ax=ax, vmin= min_gradient, vmax=max_gradient, annot=labels, fmt='', mask=mask)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()
    return gradients

def pull_out_encoder(autoencoder):
    input = autoencoder.get_layer('input').output
    latent_output = autoencoder.get_layer('latent_layer').output
    encoder = tf.keras.Model(input, latent_output, name="encoder")
    print(encoder.summary())
    return encoder

def main():
    tnfs, contig_ids_np, depth = load_data()
    feature_matrix_normalized, x_train, x_valid = preprocess_data(tnfs, depth)
    ids, contig_ids, contigid_to_binid, bin_id_to_contig_names = get_cami_data_truth(os.path.join(dataset_path, "ground_truth_with_length"))

    # Run AE
    if True:
        autoencoder = hvordan_er_det_her_ikke_måden_at_lave_en_AE_på(x_train, x_valid, no_layers=2,
                                                                     no_neurons_hidden=200, no_neurons_embedding=10,
                                                                     epochs=200, drop=False, bn=False, denoise=False)
        autoencoder.save('model_10_dims.h5')
        #encoder = autoencoder.get_layer('Encoder')
        encoder = pull_out_encoder(autoencoder)
        embeddings = encoder(feature_matrix_normalized, training=False)

        sensitivity(autoencoder, feature_matrix_normalized)
    else:
        autoencoder = tf.keras.models.load_model('my_model.h5')
        encoder = autoencoder.get_layer('Encoder')
        embeddings = encoder(feature_matrix_normalized, training=False)
        #sensitivity(autoencoder, feature_matrix_normalized)
        #subset = feature_matrix_normalized
        #reconstructed = autoencoder(subset).numpy()
        #errors = np.mean(np.abs(feature_matrix_normalized - reconstructed), axis=0)
        #mean_error = np.mean(errors)


        print("look")


    # Run tensorboard projector
    project_data(embeddings, contig_ids_np, bin_id_to_contig_names, bins_to_plot=10)

    print("test")

if __name__ == '__main__':
    main()