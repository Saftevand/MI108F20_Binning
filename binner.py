import clustering_methods
import abc
import data_processor
import tensorflow as tf
from tensorflow import keras
import numpy as np
from clustering_layer_xifeng import ClusteringLayer
from sklearn.cluster import KMeans
import sklearn
from sklearn import datasets, preprocessing
import time
import matplotlib.pyplot as plt
import cudf
import io
from cuml import TSNE
from cuml.cluster import DBSCAN
import sys
import vamb_clust
from itertools import cycle, islice
import os
from collections import Counter
from tensorboard.plugins import projector
from collections import Counter


import torch



class Binner(abc.ABC):
    def __init__(self, contig_ids, clustering_method, split_value, log_dir, feature_matrix=None, labels=None, x_train=None, x_valid=None):
        self.feature_matrix = feature_matrix
        self.bins = None
        self.contig_ids = contig_ids
        self.clustering_method = clustering_method
        self.x_train = x_train
        self.x_valid = x_valid
        self.encoder = None
        self.full_AE_train_history = None
        self.log_dir = f'{log_dir}/logs'
        self.train_start = None
        self.labels = labels

    @abc.abstractmethod
    def do_binning(self) -> [[]]:
        pass

    def get_assignments(self):
        try:
            result = np.vstack([self.contig_ids, self.bins])
        except:
            print('Could not combine contig ids with bin assignment')
            print('\nContig IDs:', self.contig_ids, '\nBins:', self.bins)
            sys.exit('\nProgram finished without results')
        return result

    def set_train_timestamp(self):
        self.train_start = int(time())

    def custom_vamb_loss(self, alpha):
        '''returns the loss function for the model'''

        def return_function(true, pred):

            print(true)
            print(pred)

            tnfs_pred = pred[:136]
            tnfs_true = true[:136]

            abundance_pred = keras.backend.log(pred[136:])
            abundance_true = true[136:]
            _, num_samples = abundance_pred.shape

            print(num_samples)

            tnfs_loss = 0
            for i in range(136):
                tnfs_loss += (tnfs_pred[i]-tnfs_true[i])**2

            abundance_loss = 0
            for i in range(num_samples):
                abundance_loss += abundance_pred[i] * abundance_true[i]

            '''              W_ab                    *      E_ab      +     W_tnf    *  E_tnf'''
            return ((1-alpha) * np.log(num_samples)) * abundance_loss + ((alpha/136) * tnfs_loss)
        return return_function


class Sequential_Binner(Binner):
    def __init__(self, split_value, contig_ids, clustering_method, feature_matrix, log_dir):
        super().__init__(contig_ids=contig_ids, clustering_method=clustering_method, feature_matrix=feature_matrix,
                         split_value=split_value, log_dir=log_dir)
        self._input_layer_size = None
        self.decoder = None
        self.full_autoencoder = None
        self.log_dir = log_dir

    def do_binning(self):
        self.set_train_timestamp()
        self.full_AE_train_history, self.full_autoencoder = self.train()
        self.bins = self.clustering_method.do_clustering(dataset=self.encoder.predict(self.feature_matrix))

    def _encoder(self):
        if self.encoder is not None:
            return self.encoder
        self._input_layer_size = len(self.x_train[1])

        stacked_encoder = keras.models.Sequential([
            keras.layers.Dense(100, activation="selu", input_shape=[self._input_layer_size, ],
                               kernel_initializer=keras.initializers.lecun_normal()),
            keras.layers.Dense(100, activation="selu", kernel_initializer=keras.initializers.lecun_normal()),
            keras.layers.Dense(300, activation="selu", kernel_initializer=keras.initializers.lecun_normal()),
            keras.layers.Dense(5, kernel_initializer=keras.initializers.lecun_normal()), ])
        self.encoder = stacked_encoder
        return stacked_encoder

    def _decoder(self):
        if self.decoder is not None:
            return self.decoder

        stacked_decoder = keras.models.Sequential([
            keras.layers.Dense(300, activation="selu", input_shape=[5],
                               kernel_initializer=keras.initializers.lecun_normal()),
            keras.layers.Dense(100, activation="selu", kernel_initializer=keras.initializers.lecun_normal()),
            keras.layers.Dense(100, activation="selu", kernel_initializer=keras.initializers.lecun_normal()),
            keras.layers.Dense(self._input_layer_size, kernel_initializer=keras.initializers.lecun_normal()), ])
        self.decoder = stacked_decoder
        return stacked_decoder

    def train(self, loss_funciton=keras.losses.mse, optimizer_=keras.optimizers.Adam(lr=0.01), number_of_epoch=100):
        log_dir = f'{self.log_dir}_train_{int(self.train_start)}'

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        stacked_ae = keras.models.Sequential([self._encoder(), self._decoder()])
        stacked_ae.compile(loss=loss_funciton,
                           optimizer=optimizer_)

        print('Training start')
        history = stacked_ae.fit(x=self.x_train, y=self.x_train, epochs=number_of_epoch,
                                 validation_data=[self.x_valid, self.x_valid], callbacks=[tensorboard_callback])
        print('Training ended')
        return history, stacked_ae

    def extract_features(self, feature_matrix):
        return self.encoder.predict(feature_matrix)


class Badass_Binner(Binner):

    def __init__(self, split_value, contig_ids, clustering_method, feature_matrix, log_dir, labels=None, x_train=None, x_valid=None):
        super().__init__(contig_ids=contig_ids, clustering_method=clustering_method, feature_matrix=feature_matrix,
                         split_value=split_value, log_dir=log_dir, labels=labels, x_train=x_train, x_valid=x_valid)
        self._input_layer_size = None
        self.decoder = None
        self.decoder = None
        self.autoencoder = None
        self.history = None
        self.log_dir = log_dir + time.strftime("%Y%m%d-%H%M%S")
        self.encoding_size = 0

    def save_autoencoder(self):
        model_json = self.autoencoder.to_json()
        with open("autoencoder.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.autoencoder.save_weights("autoencoder.h5")

        print("Saved autoencoder")

    def extract_encoder(self):
        input = self.autoencoder.get_layer('input').output
        latent_output = self.autoencoder.get_layer('latent_layer').output
        self.encoder = keras.Model(input, latent_output, name="encoder")
        print(self.encoder.summary())

    def load_autoencoder(self):
        json_file = open('autoencoder.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = keras.models.model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("autoencoder.h5")
        self.autoencoder = loaded_model
        print("Loaded autoencoder")


    def do_binning(self, load_model=False, lars_load=True):
        if not load_model:
            self.build_pretraining_model([100, 100, 200, 32], False, 4, learning_rate=0.001)  # default adam = 0.001
            self.pretrain(self.feature_matrix, self.labels, batch_size=128, epochs=400, validation_split=0.2, shuffle=True)
            self.extract_encoder()
            self.save_autoencoder()
        elif lars_load:
            self.autoencoder = tf.keras.models.load_model('model_10_dims.h5')
        else:
            self.load_autoencoder()


        '''
        encoded = self.encode(self.feature_matrix)

        fuck= torch.tensor(encoded.numpy())

        first = time.strftime('%H:%M:%S')
        print(first)

        for i in range(18000):
            vamb_clust._calc_distances_euclidean(fuck, 2)

        snd = time.strftime('%H:%M:%S')


        print(snd)
        '''
        self.extract_encoder()
        self.include_clustering_loss(learning_rate=0.0001, loss_weights=[1, 0.05])

        self.fit_dbscan(self.feature_matrix, y=self.labels, batch_size=4000, epochs=1, cuda=True)
            #self.bins = self.clustering_method.do_clustering(self.encoder.predict(self.feature_matrix))

        return self.bins

    def do_iris_binning(self):
        iris = datasets.load_iris()
        X = iris.data

        #gen noise
        dim1 = np.random.uniform(3., 8., 50)
        dim2 = np.random.uniform(1., 5., 50)
        dim3 = np.random.uniform(0., 6., 50)
        dim4 = np.random.uniform(-1., 4, 50)

        noise = np.stack([dim1, dim2, dim3, dim4], axis=1)
        X = np.vstack((X, noise))

        scaler = preprocessing.MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)

        y = iris.target

        y = np.append(y,[3 for _ in range(50)])

        # TODO lige nu er det bygget til iris data. --> kører ned til 2 dims -->  ingen tSNE
        self.build_pretraining_model([4,10, 2], True, 4, learning_rate=0.0001) #default adam = 0.001
        self.pretrain(X, y, batch_size=10, epochs=50, validation_split=0.2, shuffle=True)
        self.include_clustering_loss(learning_rate=0.0001, loss_weights=[0.5, 0.5])
        self.fit_iris(x=X, y=y, batch_size=20, epochs=5, cuda=False)


        #self.bins = self.clustering_method.do_clustering(self.encoder.predict(X))
        #return self.bins

    def build_pretraining_model(self, layers, toy_data=False, toy_dims=4, learning_rate=0.001):

        init = "glorot_uniform"
        activation = "elu"

        if toy_data:
            input_shape = toy_dims
        else:
            input_shape = self.feature_matrix.shape[1]

        if len(layers) == 1:
            encoder_input = keras.layers.Input(shape=(input_shape,), name="non_sequential")
            enc = keras.layers.Dense(layers[0], activation=activation, name=f'Encoder_Layer')(encoder_input)
            decoder_output = keras.layers.Dense(input_shape, activation="sigmoid")(enc)

            autoencoder = keras.Model(inputs=encoder_input, outputs=decoder_output, name="autoencoder")
            encoder = keras.Model(encoder_input, enc, name="encoder")

            opt = keras.optimizers.Adam(learning_rate=learning_rate)

            self.encoder = encoder
            self.autoencoder = autoencoder
            self.autoencoder.compile(optimizer=opt, loss='mse')

            print(self.autoencoder.summary())
            return

        # Encoder
        encoder_input = keras.layers.Input(shape=(input_shape,), name="input")
        enc = encoder_input

        # This layer? should we have it?
        #enc = keras.layers.Activation(activation)(encoder_input)

        layer_no = 1
        for layer in layers[:-1]:
            enc = keras.layers.Dense(layer, activation=activation, kernel_initializer=init,
                                     name=f'Encoder_Layer_{layer_no}')(enc)
            layer_no += 1

        # latent layer
        out = keras.layers.Dense(layers[-1], kernel_initializer=init, name='latent_layer')(enc)

        # create encoder
        #encoder = keras.Model(encoder_input, out, name="encoder")

        # first layer of decoder outside loop - we need to get hold of encoder_output to define encoder and extra output
        layers.reverse()
        layers.pop(0)
        print(layers)
        layer_no = 1

        for layer in layers:
            out = keras.layers.Dense(layer, activation=activation, kernel_initializer=init,
                                     name=f'Decoder_layer_{layer_no}')(out)
            layer_no += 1
        decoder_output = keras.layers.Dense(input_shape, activation="sigmoid", kernel_initializer=init)(out)


        autoencoder = keras.Model(inputs=encoder_input, outputs=decoder_output, name="autoencoder")

        #self.encoder = encoder
        self.autoencoder = autoencoder

        opt = keras.optimizers.Adam(learning_rate=learning_rate)

        # We can put loss_weights=[0.9, 0.1] into the compile method to add weights to losses - regularization
        self.autoencoder.compile(optimizer=opt, loss='mse')

        print(self.autoencoder.summary())

    def pretrain(self, x, y, batch_size=30, epochs=60, validation_split=0.2, shuffle=True):
        log_dir = f'{self.log_dir}_pretraining_{time.strftime("%Y%m%d-%H%M%S")}'

        def plot_to_image(figure):
            """Converts the matplotlib plot specified by 'figure' to a PNG image and
            returns it. The supplied figure is closed and inaccessible after this call."""
            # Save the plot to a PNG in memory.
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            # Closing the figure prevents it from being displayed directly inside
            # the notebook.
            plt.close(figure)
            buf.seek(0)
            # Convert PNG buffer to TF image
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            # Add the batch dimension
            image = tf.expand_dims(image, 0)
            return image

        def embeddings(epoch, logs):
            file_write_embeddings = tf.summary.create_file_writer(log_dir + 'pretrain_embeddings')
            # TODO we do not have to encode here. THe clustering part encodes all data at each batch. could be shared
            encoded_data = self.encoder.predict(x=x)
            # embedded = TSNE(n_components=2).fit_transform(encoded_data)
            figure = plt.figure(figsize=(10, 10))
            plt.title("Pretrain_embeddings")
            axes = plt.gca()
            #axes.set_xlim([-1.5, 1.5])
            #axes.set_ylim([-1.5, 1.5])
            # x = encoded_data[:, 0]
            # y = encoded_data[:, 1]
            # farvelade ---------------------------
            colors = np.array(list(islice(cycle(['blue', 'red', 'limegreen', 'turquoise', 'darkorange', 'magenta', 'black', 'brown']),
                                          int(max(y) + 1))))
            plt.scatter(encoded_data[:, 0], encoded_data[:, 1], s=6, color=colors[y])

            image = plot_to_image(figure)
            with file_write_embeddings.as_default():
                tf.summary.image("Pretrain_Embeddings", image, step=epoch)

        embedding_callback = keras.callbacks.LambdaCallback(on_epoch_end=embeddings)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # todo #info# Embedding callback not used!
        # begin pretraining
        self.history = self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs,
                                            validation_split=validation_split, shuffle=shuffle,
                                            callbacks=[tensorboard_callback])
        #np.save('latent_representation', self.encode(self.feature_matrix))

    def include_lars_loss(self, learning_rate=0.001, loss_weights=[0.5, 0.5]):
        #output_of_input = self.autoencoder.layers[0].output
        #output_of_input = self.autoencoder.get_layer('Encoder').get_layer('input').output
        #latent_output = self.autoencoder.get_layer('Encoder').get_layer('embedding_layer').output
        #decoder_output = self.autoencoder.layers[-1].output

        #decoder_input = self.autoencoder.get_layer('Decoder').get_layer('embedding').output
        #decoder_output = self.autoencoder.get_layer('Decoder').get_layer('Decoder_output_layer').output

        encoder = self.autoencoder.get_layer('Encoder')
        latent_output = encoder.output
        decoder = self.autoencoder.get_layer('Decoder')

        #outer_input = self.autoencoder.get_layer('Input_features').output
        outer_input = tf.keras.layers.Input(shape=(141,), name="Input_features")
        embedded_features = encoder(outer_input)
        reconstructed_features = decoder(embedded_features)

        stacked_ae = tf.keras.Model(inputs=[outer_input], outputs=[reconstructed_features, latent_output], name='new_AE')

        #rerouted_autoencoder = keras.Model(inputs=[output_of_input], outputs=[decoder_output, latent_output], name='2_outs_autoencoder')

        opt = keras.optimizers.Adam(learning_rate=learning_rate)

        def cosine_dist(y_true, y_pred):
            losses = tf.keras.losses.CosineSimilarity(axis=1)(y_true, y_pred)

            return tf.abs(losses)
        # , loss_weights=[0., 1.]
        stacked_ae.compile(optimizer=opt, loss=['mae', 'mse'], loss_weights=loss_weights)

        self.autoencoder = stacked_ae

        print(self.autoencoder.summary())

    def include_clustering_loss(self, learning_rate=0.001, loss_weights=[0.5, 0.5]):
        output_of_input = self.autoencoder.layers[0].output
        decoder_output = self.autoencoder.layers[-1].output
        latent_output = self.autoencoder.get_layer('latent_layer').output



        rerouted_autoencoder = keras.Model(inputs=[output_of_input], outputs=[decoder_output, latent_output],
                                           name='2outs_autoencoder')

        opt = keras.optimizers.Adam(learning_rate=learning_rate)

        def cosine_dist(y_true, y_pred):
            losses = tf.keras.losses.CosineSimilarity(axis=1)(y_true, y_pred)

            return tf.abs(losses)
        # , loss_weights=[0., 1.]
        rerouted_autoencoder.compile(optimizer=opt, loss=['mse', 'mse'], loss_weights=loss_weights)

        self.autoencoder = rerouted_autoencoder

        print(self.autoencoder.summary())

    def embeddings_projector(self, epoch, labels, assignments, medoids, encodings):
        metadata_path = os.path.join(self.log_dir, f'metadata_epoch_{epoch}.tsv')
        with open(metadata_path, "w") as metadata_file:
            metadata_file.write('Label\tPrediction\n')
            for label, assignment in zip(labels, assignments):
                metadata_file.write(f'{label}\t{assignment}\n')

        features = tf.Variable(encodings, name=f'Features_{epoch}')
        checkpoint = tf.train.Checkpoint(embedding=features)
        checkpoint.save(os.path.join(self.log_dir, f'embedding_epoch_{epoch}.ckpt'))

        # Set up config
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        # The name of the tensor will be suffixed by ''
        embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
        embedding.metadata_path = f'metadata_epoch_{epoch}.tsv'
        projector.visualize_embeddings(self.log_dir, config)

    def plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    def embeddings(self, epoch, labels, assignments, medoids, encodings):
        file_write_embeddings = tf.summary.create_file_writer(self.log_dir)
        # TODO we do not have to encode here. THe clustering part encodes all data at each batch. could be shared
        encoded_data = encodings

        figure = plt.figure(figsize=(10, 10))
        plt.title("Embeddings")
        axes = plt.gca()
        axes.set_xlim([-1.5, 1.5])
        axes.set_ylim([-1.5, 1.5])
        X = encoded_data

        colors = np.array(
            list(islice(cycle(['blue', 'red', 'limegreen', 'turquoise', 'darkorange', 'magenta', 'black', 'brown']),
                        int(max(labels) + 1))))
        plt.scatter(X[:, 0], X[:, 1], s=12, color=colors[labels])

        image = self.plot_to_image(figure)
        with file_write_embeddings.as_default():
            tf.summary.image("Embeddings", image, step=epoch)

        # assignments = [0,0,0,0,1,1,1,1,2,2,2,......]
        '''Save cluster embeddings'''
        # returns list with only 1 element
        largest_cluster_indices = []
        second_largest_indices = []
        small_clusters_indices = []
        third_clust_indices = []

        # index (assignment) of largest cluster
        counter = Counter(assignments).most_common(3)
        most_common_cluster = counter[0][0]
        second_most_common_cluster = counter[1][0]
        trd_cluster = counter[2][0]
        print(f'Largest cluster ID : {most_common_cluster}')

        print(assignments)
        print(most_common_cluster)
        print(medoids)

        mask = np.ones(len(medoids), dtype=bool)
        mask[most_common_cluster] = False
        mask[second_most_common_cluster] = False
        mask[trd_cluster] = False

        most_common_medoid = medoids[most_common_cluster]
        second_most_common_medoid = medoids[second_most_common_cluster]
        third_clust_medoid = medoids[trd_cluster]

        medoids_small_clusters = medoids[mask]

        for i, point in enumerate(assignments):
            # largest_cluster_indices.append(i) if point == most_common_cluster else second_largest_indices.append(i) if point == second_most_common_cluster else third_clust_medoid.append(i) if point == third_clust_medoid else small_clusters_indices.append(i)
            largest_cluster_indices.append(i) if point == most_common_cluster else second_largest_indices.append(
                i) if point == second_most_common_cluster else third_clust_indices.append(
                i) if point == trd_cluster else small_clusters_indices.append(i)

        mask = np.zeros(len(X), dtype=bool)
        mask[largest_cluster_indices] = True

        large_cluster = X[mask]

        mask[largest_cluster_indices] = False

        mask[second_largest_indices] = True

        second_cluster = X[mask]

        mask[second_largest_indices] = False

        mask[third_clust_indices] = True

        third_cluster = X[mask]

        mask[third_clust_indices] = True
        mask[second_largest_indices] = True
        mask[largest_cluster_indices] = True

        small_clusters = X[~mask]

        file_write_embeddings = tf.summary.create_file_writer(self.log_dir)
        # TODO we do not have to encode here. THe clustering part encodes all data at each batch. could be shared
        # encoded_data = self.encoder.predict(x=x)
        # embedded = TSNE(n_components=2).fit_transform(encoded_data)
        figure = plt.figure(figsize=(10, 10))
        plt.title("Cluster_embeddings")
        axes = plt.gca()
        axes.set_xlim([-1.5, 1.5])
        axes.set_ylim([-1.5, 1.5])
        # x = encoded_data[:, 0]
        # y = encoded_data[:, 1]
        X = encoded_data
        # farvelade ---------------------------

        medoid_labels = [i for i, c in enumerate(medoids)]

        colors = np.array(list(islice(cycle(['black']), len(medoids))))
        print(colors)
        print(medoids)
        print(f'X: {X[0]}')
        x, y = medoids[:, 0], medoids[:, 1]
        print(x)
        print(y)

        plt.scatter(medoids_small_clusters[:, 0], medoids_small_clusters[:, 1], s=600, color='black', marker=".")
        plt.scatter(small_clusters[:, 0], small_clusters[:, 1], s=50, color='darkorange', marker="x")

        # plot largest cluster
        plt.scatter(most_common_medoid[0], most_common_medoid[1], s=600, color='red', marker=".")
        plt.scatter(large_cluster[:, 0], large_cluster[:, 1], s=50, color='blue', marker="x")

        # plot second largest cluster
        plt.scatter(second_most_common_medoid[0], second_most_common_medoid[1], s=600, color='limegreen', marker=".")
        plt.scatter(second_cluster[:, 0], second_cluster[:, 1], s=50, color='darkgreen', marker="x")

        # plot 3rd largest cluster
        plt.scatter(third_clust_medoid[0], third_clust_medoid[1], s=600, color='magenta', marker=".")
        plt.scatter(third_cluster[:, 0], third_cluster[:, 1], s=50, color='gray', marker="x")

        '''
        medoid_labels = [i for i, c in enumerate(medoids)]


        colors = np.array(
            list(islice(cycle(['blue', 'red', 'limegreen', 'turquoise', 'darkorange', 'magenta', 'black', 'brown']),
                        int(max(medoid_labels) + 1))))
        plt.scatter(medoids[:, 0], medoids[:, 1], s=600, color=colors[medoid_labels], marker=".")

        # plt.scatter(x, y, s=2, cmap=plt.cm.get_cmap("jet", 256))

        colors = np.array(
            list(islice(cycle(['blue', 'red', 'limegreen', 'turquoise', 'darkorange', 'magenta', 'black', 'brown']),
                        int(max(assignments) + 1))))
        plt.scatter(X[:, 0], X[:, 1], s=50, color=colors[assignments], marker="x")
        '''

        image = self.plot_to_image(figure)
        with file_write_embeddings.as_default():
            tf.summary.image("Cluster_Embeddings", image, step=epoch)

    def tsne_embeddings(self, epoch, encoded_data, labels, assignments, medoids, logs=None):
        embedding_dir = self.log_dir + 'embeddings'
        file_write_embeddings = tf.summary.create_file_writer(embedding_dir)
        # encoded_data = self.encoder.predict(x=self.feature_matrix)

        embedded = TSNE(n_components=2).fit_transform(encoded_data)

        figure = plt.figure(figsize=(10, 10))
        plt.title("Embeddings")

        plt.scatter(embedded[:, 0], embedded[:, 1], s=50)

        image = self.plot_to_image(figure)
        with file_write_embeddings.as_default():
            tf.summary.image("Embeddings", image, step=epoch)

    def fit_iris(self, x, y=None, batch_size=5, epochs=30, cuda=False):
        labels = y
        tensorboard = keras.callbacks.TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=0,
            batch_size=batch_size,
            write_graph=True,
            write_grads=True
        )

        data = np.copy(x)
        indices = np.arange(len(data))

        no_batches = len(x) / batch_size

        # 1. get appropriate batch

        '''Next 15 lines regarding tensorboard and callbacks are from Stack overflow user: "erenon"
           https://stackoverflow.com/questions/44861149/keras-use-tensorboard-with-train-on-batch'''
        # create tf callback

        # embeddings_callback = keras.callbacks.LambdaCallback(on_epoch_end=tsne_embeddings)
        tensorboard.set_model(self.autoencoder)

        # Transform train_on_batch return value
        # to dict expected by on_batch_end callback
        def named_logs(model, logs):
            result = {}
            for l in zip(model.metrics_names, logs):
                result[l[0]] = l[1]
            return result

        for i in range(epochs):

            np.random.RandomState(i).shuffle(data)
            np.random.RandomState(i).shuffle(indices)
            np.random.RandomState(i).shuffle(y)
            batches = np.array_split(data, no_batches)
            batch_indices_list = np.array_split(indices, no_batches)
            amount_batches = len(batches)

            print(f'Current epoch: {i + 1} of total epochs: {epochs}')
            total_loss = 0
            reconstruct_loss = 0
            cluster_loss = 0
            for batch_no, batch in enumerate(batches):
                batch_indices = batch_indices_list[batch_no]
                # print(f'Current batch: {batch_no + 1} of total batches: {amount_batches}')

                # 2. encode all data
                encoded_data_full = self.encode(data).numpy()

                # 3. cluster

                contig_medoid_assignment = np.zeros(encoded_data_full.shape)

                # Kan gøres bedre, assigne mens, clusters bliver lavet

                cluster_generator = vamb_clust.cluster(encoded_data_full, cuda=cuda)

                clusters = ((encoded_data_full[c.medoid], c.members) for (i, c) in enumerate(cluster_generator))

                # X (original matrix)
                no_clusters = 0
                intermediate_clusters = np.zeros(len(encoded_data_full),dtype=int)
                medoids = []

                for medoid, members in clusters:
                    medoids.append(medoid)

                    for member in members:
                        contig_medoid_assignment[member] = medoid
                        intermediate_clusters[member] = no_clusters
                    no_clusters += 1
                print(intermediate_clusters)

                print(f'No. clusters: {no_clusters+1}')
                np_medoids = np.array(medoids)


                '''
                encoded_batch = self.encode(batch).numpy()

                kmeans = KMeans(n_clusters=3, random_state=0).fit(encoded_data_full)
                batch_predictions = kmeans.predict(encoded_batch)
                full_predictions = kmeans.predict(encoded_data_full)
                centroids = kmeans.cluster_centers_

                index_ = 0
                truths = np.empty((batch_size, 2))

                for centr in centroids:
                    for q, j in enumerate(batch_predictions):
                        if j == index_:
                            truths[q] = centr
                    index_ += 1

                # TODO centroids er ikke punkter!
                list_of_losses = self.autoencoder.train_on_batch(x=[batch], y=[batch, truths])
                '''

                truth_medoids = tf.gather(params=contig_medoid_assignment, indices=batch_indices)
                # print(f'HVA SKER DER?!: {truth_medoids}')

                list_of_losses = self.autoencoder.train_on_batch(x=[batch], y=[batch, truth_medoids])
                # print(self.autoencoder.metrics_names)
                # print(list_of_losses)

                total_loss += list_of_losses[0]
                reconstruct_loss += list_of_losses[1]
                cluster_loss += list_of_losses[2]

            # var = [(j, i) for j, i in sorted(zip(full_predictions, encoded_data_full), key=lambda pair: pair[0])]
            # sorted_cluster_assignments_full = [i[0] for i in var]
            # sorted_encodings = np.array([i[1] for i in var])

            history_obj = list()
            tot_avg = total_loss / amount_batches
            history_obj.append(tot_avg)

            recon_avg = reconstruct_loss / amount_batches
            history_obj.append(recon_avg)

            clust_avg = cluster_loss / amount_batches
            history_obj.append(clust_avg)

            print(f'Total loss: {tot_avg}, \t Reconst loss: {recon_avg}, \t Clust loss: {clust_avg}')

            self.embeddings(i + 1, labels=labels, assignments=intermediate_clusters, medoids=np_medoids, encodings=encoded_data_full)

            self.embeddings_projector(i + 1, labels=labels, assignments=intermediate_clusters, medoids=np_medoids, encodings=encoded_data_full)
            tensorboard.on_epoch_end(i + 1, named_logs(self.autoencoder, history_obj))

        tensorboard.on_train_end(None)

    def fit_iris_kmeans(self, x, y=None, batch_size=5, epochs=30):
        labels = y

        timestr = time.strftime("%Y%m%d-%H%M%S")
        logdir = self.log_dir + '_IRIS_' + timestr
        tensorboard = keras.callbacks.TensorBoard(
            log_dir=logdir,
            histogram_freq=0,
            batch_size=batch_size,
            write_graph=True,
            write_grads=True
        )

        def plot_to_image(figure):
            """Converts the matplotlib plot specified by 'figure' to a PNG image and
            returns it. The supplied figure is closed and inaccessible after this call."""
            # Save the plot to a PNG in memory.
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            # Closing the figure prevents it from being displayed directly inside
            # the notebook.
            plt.close(figure)
            buf.seek(0)
            # Convert PNG buffer to TF image
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            # Add the batch dimension
            image = tf.expand_dims(image, 0)
            return image

        def embeddings(epoch, labels, assignments, medoids, encodings):
            file_write_embeddings = tf.summary.create_file_writer(logdir + 'embeddings')
            # TODO we do not have to encode here. THe clustering part encodes all data at each batch. could be shared
            encoded_data = encodings

            figure = plt.figure(figsize=(12, 12))
            plt.title("Embeddings")
            axes = plt.gca()
            axes.set_xlim([-1.5, 1.5])
            axes.set_ylim([-1.5, 1.5])
            X = encoded_data

            colors = np.array(list(islice(cycle(['blue', 'red', 'limegreen', 'turquoise', 'darkorange', 'magenta', 'black', 'brown']),
                                          int(max(labels) + 1))))
            plt.scatter(X[:, 0], X[:, 1], s=12, color=colors[labels])


            image = plot_to_image(figure)
            with file_write_embeddings.as_default():
                tf.summary.image("Embeddings", image, step=epoch)



            '''Save cluster embeddings'''
            file_write_embeddings = tf.summary.create_file_writer(logdir + 'cluster_embeddings')
            # TODO we do not have to encode here. THe clustering part encodes all data at each batch. could be shared
            #encoded_data = self.encoder.predict(x=x)
            # embedded = TSNE(n_components=2).fit_transform(encoded_data)
            figure = plt.figure(figsize=(12, 12))
            plt.title("Cluster_embeddings")
            axes = plt.gca()
            axes.set_xlim([-1.5,1.5])
            axes.set_ylim([-1.5, 1.5])
            # x = encoded_data[:, 0]
            # y = encoded_data[:, 1]
            X = encoded_data
            # farvelade ---------------------------
            medoid_labels = [i for i, c in enumerate(medoids)]

            colors = np.array(
                list(islice(cycle(['blue', 'red', 'limegreen', 'turquoise', 'darkorange', 'magenta', 'black', 'brown']),
                            int(max(medoid_labels) + 1))))
            plt.scatter(medoids[:, 0], medoids[:, 1], s=600, color=colors[medoid_labels], marker=".")


            # plt.scatter(x, y, s=2, cmap=plt.cm.get_cmap("jet", 256))

            colors = np.array(
                list(islice(cycle(['blue', 'red', 'limegreen', 'turquoise', 'darkorange', 'magenta', 'black', 'brown']),
                            int(max(assignments) + 1))))
            plt.scatter(X[:, 0], X[:, 1], s=50, color=colors[assignments], marker="x")

            image = plot_to_image(figure)

            with file_write_embeddings.as_default():
                tf.summary.image("Cluster_Embeddings", image, step=epoch)




        data = np.copy(x)
        indices = np.arange(len(data))

        no_batches = len(x) / batch_size

        # 1. get appropriate batch

        '''Next 15 lines regarding tensorboard and callbacks are from Stack overflow user: "erenon"
           https://stackoverflow.com/questions/44861149/keras-use-tensorboard-with-train-on-batch'''
        # create tf callback

        # embeddings_callback = keras.callbacks.LambdaCallback(on_epoch_end=tsne_embeddings)
        tensorboard.set_model(self.autoencoder)

        # Transform train_on_batch return value
        # to dict expected by on_batch_end callback
        def named_logs(model, logs):
            result = {}
            for l in zip(model.metrics_names, logs):
                result[l[0]] = l[1]
            return result

        for i in range(epochs):


            np.random.RandomState(i).shuffle(data)
            np.random.RandomState(i).shuffle(indices)
            np.random.RandomState(i).shuffle(y)
            batches = np.array_split(data, no_batches)
            batch_indices_list = np.array_split(indices, no_batches)
            amount_batches = len(batches)

            print(f'Current epoch: {i + 1} of total epochs: {epochs}')
            total_loss = 0
            reconstruct_loss = 0
            cluster_loss = 0
            for batch_no, batch in enumerate(batches):
                batch_indices = batch_indices_list[batch_no]
                #print(f'Current batch: {batch_no + 1} of total batches: {amount_batches}')

                # 2. encode all data
                encoded_data_full = self.encode(data).numpy()

                # 3. cluster
                '''
                assigned_medoids_per_contig = np.zeros(encoded_data_full.shape)

                # Kan gøres bedre, assigne mens, clusters bliver lavet
                
                cluster_generator = vamb_clust.cluster(encoded_data_full)

                clusters = ((encoded_data_full[c.medoid], c.members) for (i, c) in enumerate(cluster_generator))


                no_clusters = 0
                intermediate_clusters = []
                medoids = []

                for medoid, members in clusters:
                    medoids.append(medoid)
                    for member in members:
                        assigned_medoids_per_contig[member] = medoid
                        intermediate_clusters.append(no_clusters)
                    no_clusters += 1

                print(f'No. clusters: {no_clusters}')
                np_medoids = np.array(medoids)

                
                '''
                encoded_batch = self.encode(batch).numpy()

                kmeans = KMeans(n_clusters=3, random_state=0).fit(encoded_data_full)
                batch_predictions = kmeans.predict(encoded_batch)
                full_predictions = kmeans.predict(encoded_data_full)
                centroids = kmeans.cluster_centers_

                index_ = 0
                truths = np.empty((batch_size, 2))


                for centr in centroids:
                    for q, j in enumerate(batch_predictions):
                        if j == index_:
                            truths[q] = centr
                    index_ += 1

                #TODO centroids er ikke punkter!
                list_of_losses = self.autoencoder.train_on_batch(x=[batch], y=[batch, truths])





                #truth_medoids = tf.gather(params=assigned_medoids_per_contig, indices=batch_indices)
                #print(f'HVA SKER DER?!: {truth_medoids}')

                #list_of_losses = self.autoencoder.train_on_batch(x=[batch], y=[batch, truth_medoids])
                # print(self.autoencoder.metrics_names)
                # print(list_of_losses)

                total_loss += list_of_losses[0]
                reconstruct_loss += list_of_losses[1]
                cluster_loss += list_of_losses[2]

            #var = [(j, i) for j, i in sorted(zip(full_predictions, encoded_data_full), key=lambda pair: pair[0])]
            #sorted_cluster_assignments_full = [i[0] for i in var]
            #sorted_encodings = np.array([i[1] for i in var])

            history_obj = list()
            tot_avg = total_loss / amount_batches
            history_obj.append(tot_avg)

            recon_avg = reconstruct_loss / amount_batches
            history_obj.append(recon_avg)

            clust_avg = cluster_loss / amount_batches
            history_obj.append(clust_avg)

            print(f'Total loss: {tot_avg}, \t Reconst loss: {recon_avg}, \t Clust loss: {clust_avg}')

            embeddings(i + 1, labels=labels, assignments=full_predictions, medoids=centroids, encodings=encoded_data_full)
            #embeddings(i + 1, labels=labels, assignments=intermediate_clusters, medoids=np_medoids)
            tensorboard.on_epoch_end(i + 1, named_logs(self.autoencoder, history_obj))

        tensorboard.on_train_end(None)

    def build_model(self, layers, toy_data=False, toy_size=4, learning_rate=0.001):

        self.encoding_size = 50

        init = "glorot_uniform"
        activation = "elu"

        if toy_data:
            input_shape = toy_size
        else:
            input_shape = self.feature_matrix.shape[1]

        #Encoder
        encoder_input = keras.layers.Input(shape=(input_shape,), name="non_sequential")

        # This layer? should we have it?
        enc = keras.layers.Activation("elu")(encoder_input)


        layer_no = 1
        for layer in layers[:-1]:
            enc = keras.layers.Dense(layer, activation=activation, kernel_initializer=init, name=f'Encoder_Layer_{layer_no}')(enc)
            layer_no += 1

        encoder_output = keras.layers.Dense(layers[-1], kernel_initializer=init, name='latent_layer')(enc)



        encoder = keras.Model(encoder_input, encoder_output, name="encoder")

        # first layer of decoder outside loop - we need to get hold of encoder_output to define encoder and extra output
        layers.reverse()
        layers.pop(0)
        print(layers)
        layer_no = 1
        dec = keras.layers.Dense(layers[0], activation=activation, kernel_initializer=init, name=f'Decoder_layer_{layer_no}')(encoder_output)
        layers.pop(0)
        print(layers)
        layer_no += 1

        for layer in layers:
                dec = keras.layers.Dense(layer, activation=activation, kernel_initializer=init, name=f'Decoder_layer_{layer_no}')(dec)
        decoder_output = keras.layers.Dense(input_shape, activation="sigmoid", kernel_initializer=init)(dec)

        #dec1 = keras.layers.Dense(100, activation=activation, kernel_initializer=init)(encoder_output)
        #dec2 = keras.layers.Dense(100, activation=activation, kernel_initializer=init)(dec1)

        #decoder_output = keras.layers.Dense(input_shape, activation="sigmoid", kernel_initializer=init)(dec)


        autoencoder = keras.Model(inputs=[encoder_input], outputs=[decoder_output, encoder_output], name="autoencoder")

        self.encoder = encoder
        self.autoencoder = autoencoder

        opt = keras.optimizers.Adam(learning_rate=learning_rate)

        def cosine_dist(y_true, y_pred):
            losses = tf.keras.losses.CosineSimilarity(axis=1)(y_true, y_pred)

            return tf.abs(losses)

        # We can put loss_weights=[0.9, 0.1] into the compile method to add weights to losses - regularization
        self.autoencoder.compile(optimizer=opt, loss=['mse', cosine_dist])

        print(self.autoencoder.summary())
        print(self.encoder.summary())

    def train(self, batch_size=400, epochs=100, validation_split=0.2, shuffle=True):

        log_dir = f'{self.log_dir}_{int(time())}'

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # We dont have target labels for latent encodings. They are calculated in loss func. Dummy is given instead
        dummy_matrix = np.arrange(len(self.feature_matrix))

        # begin pretraining
        self.history = self.autoencoder.fit(self.feature_matrix, [self.feature_matrix, dummy_matrix], batch_size=batch_size,
                                            epochs=epochs, validation_split=validation_split, shuffle=shuffle,
                                            callbacks=[tensorboard_callback])
        np.save('latent_representation', self.encode(self.feature_matrix))

    def encode(self, inp):
        return self.encoder(inp)

    def decode(self, input):
        return self.decoder(input)

    def fit(self, x, y=None, batch_size=4000, epochs=10, cuda=True):

        tensorboard = keras.callbacks.TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=0,
            batch_size=batch_size,
            write_graph=True,
            write_grads=True
        )

        data = np.copy(x)
        labels = y
        indices = np.arange(len(data))

        no_batches = len(x)/batch_size

        # 1. get appropriate batch


        '''Next 15 lines regarding tensorboard and callbacks are from Stack overflow user: "erenon"
           https://stackoverflow.com/questions/44861149/keras-use-tensorboard-with-train-on-batch'''
        # create tf callback

        #embeddings_callback = keras.callbacks.LambdaCallback(on_epoch_end=tsne_embeddings)
        tensorboard.set_model(self.autoencoder)

        # Transform train_on_batch return value
        # to dict expected by on_batch_end callback
        def named_logs(model, logs):
            result = {}
            for l in zip(model.metrics_names, logs):
                result[l[0]] = l[1]
            return result

        for i in range(epochs):

            np.random.RandomState(i).shuffle(data)
            np.random.RandomState(i).shuffle(indices)
            np.random.RandomState(i).shuffle(labels)
            batches = np.array_split(data, no_batches)
            batch_indices_list = np.array_split(indices, no_batches)
            amount_batches = len(batches)

            print(f'Current epoch: {i+1} of total epochs: {epochs}')
            total_loss = 0
            reconstruct_loss = 0
            cluster_loss = 0
            for batch_no, batch in enumerate(batches):
                batch_indices = batch_indices_list[batch_no]
                print(f'Current batch: {batch_no+1} of total batches: {amount_batches}')

                # 2. encode all data
                encoded_data_full = self.encode(data).numpy()

                # 3. cluster
                assigned_medoids_per_contig = np.zeros(encoded_data_full.shape)
                assignments = np.zeros(len(encoded_data_full))
                # Kan gøres bedre, assigne mens, clusters bliver lavet

                cluster_generator = vamb_clust.cluster(encoded_data_full, cuda=cuda)

                clusters = ((encoded_data_full[c.medoid], c.members) for (i, c) in enumerate(cluster_generator))
                centroids = []
                no_clusters = 0
                for medoid, members in clusters:
                    centroid = np.mean(tf.gather(params=encoded_data_full, indices=members), axis=0)
                    centroids.append(centroid)
                    for member in members:
                        assigned_medoids_per_contig[member] = centroid
                        assignments[member] = no_clusters
                    no_clusters += 1

                print(f'No. clusters: {no_clusters}')

                truth_medoids = tf.gather(params=assigned_medoids_per_contig, indices=batch_indices)

                list_of_losses = self.autoencoder.train_on_batch(x=[batch], y=[batch, truth_medoids])
                # print(self.autoencoder.metrics_names)
                # print(list_of_losses)

                total_loss += list_of_losses[0]
                reconstruct_loss += list_of_losses[1]
                cluster_loss += list_of_losses[2]
                print(vamb_clust.VAMB_POO)

            history_obj = list()
            tot_avg = total_loss / amount_batches
            history_obj.append(tot_avg)

            recon_avg = reconstruct_loss / amount_batches
            history_obj.append(recon_avg)

            clust_avg = cluster_loss / amount_batches
            history_obj.append(clust_avg)

            print(f'Total loss: {tot_avg}, \t Reconst loss: {recon_avg}, \t Clust loss: {clust_avg}')

            self.tsne_embeddings(epoch=i+1,encoded_data=encoded_data_full, labels=labels, assignments=assignments, medoids=centroids)

            tensorboard.on_epoch_end(i+1, named_logs(self.autoencoder, history_obj))
        self.embeddings_projector(epoch=i+1, labels=labels, assignments=assignments, medoids=centroids,
                                  encodings=encoded_data_full)
        tensorboard.on_train_end(None)

    def fit_dbscan(self, x, y=None, batch_size=4000, epochs=10, cuda=True):

        tensorboard = keras.callbacks.TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=0,
            batch_size=batch_size,
            write_graph=True,
            write_grads=True
        )

        data = np.copy(x)
        labels = y
        indices = np.arange(len(data))

        no_batches = len(x)/batch_size

        # 1. get appropriate batch


        '''Next 15 lines regarding tensorboard and callbacks are from Stack overflow user: "erenon"
           https://stackoverflow.com/questions/44861149/keras-use-tensorboard-with-train-on-batch'''
        # create tf callback

        #embeddings_callback = keras.callbacks.LambdaCallback(on_epoch_end=tsne_embeddings)
        tensorboard.set_model(self.autoencoder)

        # Transform train_on_batch return value
        # to dict expected by on_batch_end callback
        def named_logs(model, logs):
            result = {}
            for l in zip(model.metrics_names, logs):
                result[l[0]] = l[1]
            return result

        for i in range(epochs):

            np.random.RandomState(i).shuffle(data)
            np.random.RandomState(i).shuffle(indices)
            np.random.RandomState(i).shuffle(labels)
            batches = np.array_split(data, no_batches)
            batch_indices_list = np.array_split(indices, no_batches)
            amount_batches = len(batches)

            print(f'Current epoch: {i+1} of total epochs: {epochs}')
            total_loss = 0
            reconstruct_loss = 0
            cluster_loss = 0
            for batch_no, batch in enumerate(batches):
                batch_indices = batch_indices_list[batch_no]
                print(f'Current batch: {batch_no+1} of total batches: {amount_batches}')

                # 2. encode all data
                encoded_data_full = self.encode(data).numpy()
                print(encoded_data_full[:4])
                # 3. cluster
                assigned_medoids_per_contig = np.zeros(encoded_data_full.shape)
                assignments = np.zeros(len(encoded_data_full))
                first = time.strftime('%H:%M:%S')
                print(first)

                dbscan_instance = sklearn.cluster.DBSCAN(eps=2, min_samples=2)

                # dbscan_instance.fit(encoded_data_full)
                dbscan_instance.fit(encoded_data_full)

                # Todo this should be np array
                all_assignments = np.array(dbscan_instance.labels_)

                #mask = np.zeros(len(encoded_data_full), dtype=bool)
                #mask[batch_indices_list] = True
                batch_assignments = np.empty(len(batch_indices))
                for count, q in enumerate(batch_indices):
                    batch_assignments[count] = all_assignments[q]
                # batch_assignments = tf.gather(params=all_assignments, indices=batch_indices)


                batch_centroids = []

                set_batch_assignments = set(batch_assignments)
                centroid_dict = {}
                '''
                for cluster_label in set_batch_assignments:
                    cluster_mask = (all_assignments == cluster_label)
                    encoded_cluster = encoded_data_full[cluster_mask]
                    centroid = np.mean(encoded_cluster, axis=0)
                    centroid_dict[cluster_label] = centroid

                for cluster_label in batch_assignments:
                    batch_centroids.append(centroid_dict[cluster_label])
                '''

                for cluster_label in set(all_assignments):
                    # if outlier
                    if cluster_label == -1:
                        continue

                    cluster_mask = (all_assignments == cluster_label)
                    encoded_cluster = encoded_data_full[cluster_mask]
                    centroid = np.mean(encoded_cluster, axis=0)
                    centroid_dict[cluster_label] = centroid


                centroids = np.vstack(list(centroid_dict.values()))

                for index, cluster_label in zip(batch_indices, batch_assignments):
                    if cluster_label == -1:
                        dists = np.sqrt(np.sum(((centroids - encoded_data_full[index]) ** 2), axis=1))
                        shortest_dist = np.argmin(dists)
                        batch_centroids.append(centroid_dict[shortest_dist])
                        all_assignments[index] = shortest_dist
                    else:
                        batch_centroids.append(centroid_dict[cluster_label])

                first = time.strftime('%H:%M:%S')
                print(first)





                '''
                # Kan gøres bedre, assigne mens, clusters bliver lavet

                cluster_generator = vamb_clust.cluster(encoded_data_full, cuda=cuda)

                clusters = ((encoded_data_full[c.medoid], c.members) for (i, c) in enumerate(cluster_generator))
                centroids = []
                no_clusters = 0
                for medoid, members in clusters:
                    centroid = np.mean(tf.gather(params=encoded_data_full, indices=members), axis=0)
                    centroids.append(centroid)
                    for member in members:
                        assigned_medoids_per_contig[member] = centroid
                        assignments[member] = no_clusters
                    no_clusters += 1
                '''
                print(Counter(all_assignments))
                no_clusters = max(all_assignments)
                print(f'No. clusters: {no_clusters}')

                #truth_medoids = tf.gather(params=assigned_medoids_per_contig, indices=batch_indices)

                list_of_losses = self.autoencoder.train_on_batch(x=[batch], y=[batch, np.array(batch_centroids)])
                # print(self.autoencoder.metrics_names)
                # print(list_of_losses)

                total_loss += list_of_losses[0]
                reconstruct_loss += list_of_losses[1]
                cluster_loss += list_of_losses[2]
                print(vamb_clust.VAMB_POO)

            history_obj = list()
            tot_avg = total_loss / amount_batches
            history_obj.append(tot_avg)

            recon_avg = reconstruct_loss / amount_batches
            history_obj.append(recon_avg)

            clust_avg = cluster_loss / amount_batches
            history_obj.append(clust_avg)

            print(f'Total loss: {tot_avg}, \t Reconst loss: {recon_avg}, \t Clust loss: {clust_avg}')

            #self.tsne_embeddings(epoch=i+1,encoded_data=encoded_data_full, labels=labels, medoids=None, assignments=None)

            tensorboard.on_epoch_end(i+1, named_logs(self.autoencoder, history_obj))
        #self.embeddings_projector(epoch=i+1, labels=labels, assignments=all_assignments, medoids=None, encodings=encoded_data_full)
        tensorboard.on_train_end(None)
        dbscan_instance = sklearn.cluster.DBSCAN(eps=0.5, min_samples=10)

        print("Final clustering commencing... please wait...")
        enc = self.encode(self.feature_matrix)
        dbscan_instance.fit(enc)
        self.bins = np.array(dbscan_instance.labels_)
        print("Binning complete.")



class Sequential_Binner1(Binner):

    def __init__(self, split_value, contig_ids, clustering_method, feature_matrix, log_dir):
        super().__init__(contig_ids=contig_ids, clustering_method=clustering_method, feature_matrix=feature_matrix,
                         split_value=split_value, log_dir=log_dir)
        self._input_layer_size = None
        self.decoder = None
        self.decoder = None
        self.autoencoder = None
        self.history = None
        self.log_dir = f'{self.log_dir}/Sequential_binner'
        self.build_model()

    def do_binning(self):
        self.train()
        self.bins = self.clustering_method.do_clustering(self.encoder.predict(self.feature_matrix))
        return self.bins

    def build_model(self):
        n1 = 100
        n2 = 100
        n3 = 50
        n4 = 10

        init = "glorot_uniform"
        activation = "elu"

        latent_factors = 32
        input_shape = self.feature_matrix.shape[1]

        #Encoder
        encoder_input = keras.layers.Input(shape=(input_shape,), name="non_sequential")
        elu_activate = keras.layers.Activation("elu")(encoder_input)

        enc2 = keras.layers.Dense(100, activation=activation, kernel_initializer=init)(elu_activate)
        enc3 = keras.layers.Dense(100, activation=activation, kernel_initializer=init)(enc2)

        encoder_output = keras.layers.Dense(50, kernel_initializer=init)(enc3)

        dec1 = keras.layers.Dense(100, activation=activation, kernel_initializer=init)(encoder_output)
        dec2 = keras.layers.Dense(100, activation=activation, kernel_initializer=init)(dec1)

        decoder_output = keras.layers.Dense(input_shape, activation="sigmoid", kernel_initializer=init)(dec2)

        encoder = keras.Model(encoder_input, encoder_output, name="encoder")
        autoencoder = keras.Model(encoder_input, decoder_output, name="autoencoder")

        self.encoder = encoder
        self.autoencoder = autoencoder

        print(self.autoencoder.summary())

    def train(self):
        opt = keras.optimizers.Adam(learning_rate=0.001)
        #nadam = keras.optimizers.Nadam(learning_rate=0.0001)
        self.autoencoder.compile(optimizer=opt, loss='mse')

        log_dir = f'{self.log_dir}_{int(time())}'

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # begin pretraining

        self.history = self.autoencoder.fit(self.feature_matrix, self.feature_matrix, batch_size=400, epochs=1000,
                                                          validation_split=0.2, shuffle=True,
                                                          callbacks=[tensorboard_callback])
        np.save('latent_representation', self.encode(self.feature_matrix))

    def encode(self, input):
        return self.encoder(input)

    def decode(self, input):
        return self.decoder(input)


class DEC(Binner):
    def __init__(self, split_value, contig_ids, feature_matrix, clustering_method, log_dir):
        super().__init__(split_value=split_value, contig_ids=contig_ids, feature_matrix=feature_matrix,
                         clustering_method=clustering_method, log_dir=log_dir)
        self.model = None
        self.autoencoder = None
        self.n_clusters = None
        self.cluster_loss_list = None

    def do_binning(self) -> [[]]:
        pass

    def predict(self, x):  # predict cluster labels using the output of clustering layer
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def fit(self, x, y=None, maxiter=2e4, batch_size=258, tolerance_threshold=1e-3, update_interval=150):

        loss_list = []

        # tolerance threshold ~~ stopping criterion

        print('Update interval', update_interval)

        save_interval = int(x.shape[0] / batch_size) * 5  # 5 epochs
        print('Save interval', save_interval)

        # Step 1: initialize cluster centers using k-means
        t1 = time()
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(self.encoder.predict(x))
        y_pred_last = np.copy(y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        # Tensorboard setup - run board manuel
        log_dir = f'{self.log_dir}_cluster_training_{self.train_start}'
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, batch_size=batch_size)
        tensorboard_callback.set_model(self.model)

        def named_logs(model, logs):
            result = {}
            for l in zip(model.metrics_names, logs):
                result[l[0]] = l[1]
            return result

        # Step 2: deep clustering
        index = 0
        index_array = np.arange(x.shape[0])
        for ite in range(int(maxiter)):
            print(ite)
            if ite % update_interval == 0:
                q = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)

                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < tolerance_threshold:
                    print('delta_label ', delta_label, '< tol ', tolerance_threshold)
                    print('Reached tolerance threshold. Stopping training.')
                    break
            # Commented out by Xifeng himself
            # train on batch
            # if index == 0:
            #     np.random.shuffle(index_array)

            idx = index_array[index * batch_size: min((index + 1) * batch_size, x.shape[0])]
            loss = self.model.train_on_batch(x=x[idx], y=p[idx])

            # creates log for train_on_batch
            tensorboard_callback.on_epoch_end(ite, named_logs(self.model, [loss]))

            #saving loss for display later
            loss_list.append(loss)

            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

            '''
            # save intermediate model
            if ite % save_interval == 0:
                print('saving model to:', save_dir + '/DEC_model_' + str(ite) + '.h5')
                self.model.save_weights(save_dir + '/DEC_model_' + str(ite) + '.h5')
            '''
            ite += 1

        # is this necessary?
        tensorboard_callback.on_train_end(None)

        # save the trained model
        # logfile.close()
        # print('saving model to:', save_dir + '/DEC_model_final.h5')
        # self.model.save_weights(save_dir + '/DEC_model_final.h5')
        return y_pred, loss_list


class Greedy_pretraining_DEC(DEC):
    def __init__(self, split_value, clustering_method, feature_matrix, contig_ids, log_dir):
        super().__init__(split_value=split_value, clustering_method=clustering_method, feature_matrix=feature_matrix,
                         contig_ids=contig_ids, log_dir=log_dir)
        self.layers_history = []
        self.log_dir = f'{self.log_dir}/GREEDY_DEC'

    def do_binning(self, init='glorot_uniform', pretrain_optimizer=keras.optimizers.Adam(learning_rate=0.001),
                   n_clusters=10, update_interval=140, pretrain_epochs=10, finetune_epochs=100, batch_size=128,
                   save_dir='results', tolerance_threshold=1e-3, max_iterations=200, true_bins=None,
                   neuron_list=[500, 500, 2000, 10], verbose=1, pretrain_loss='mean_squared_error'):
        self.set_train_timestamp()
        self.n_clusters = n_clusters

        # layerwise and finetuned encoder
        self.greedy_pretraining(loss_function=pretrain_loss, pretrain_epochs=pretrain_epochs,
                                finetune_epochs=finetune_epochs, pretrain_optimizer=pretrain_optimizer, init=init,
                                neuron_list=neuron_list, verbose=verbose)

        # Insert clustering layer using KLD error
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output)
        self.model = keras.models.Model(inputs=self.encoder.input, outputs=clustering_layer)

        # Should be SGD according to DEC paper -S
        self.model.compile(optimizer=keras.optimizers.SGD(lr=0.01), loss='kld')

        # Special fit method defined in DEC class
        y_pred, self.cluster_loss_list = self.fit(self.feature_matrix, y=None, tolerance_threshold=tolerance_threshold,
                                                  maxiter=max_iterations, batch_size=batch_size,
                                                  update_interval=update_interval)
        self.bins = y_pred

    def greedy_pretraining(self, pretrain_optimizer, loss_function='mean_squared_error', lr=0.01, finetune_epochs=10,
                           pretrain_epochs=10, neuron_list=[500, 500, 2000, 10], input_shape=None, dropout_rate=0.2,
                           verbose=1, activation='relu', init='glorot_uniform'):
        if input_shape is None:
            # feature vector size
            input_shape = self.x_train.shape[1]

        print(f'Adding and training first layer for {pretrain_epochs} epochs')

        neurons_first_layer = neuron_list.pop(0)

        # Create first Encoder decoder pair
        input = keras.layers.Input(shape=(input_shape,))
        dropout_out = keras.layers.Dropout(dropout_rate)(input)
        enc_layer = keras.layers.Dense(neurons_first_layer, activation=activation, input_shape=[input_shape],
                                       kernel_initializer=init)
        enc_out = enc_layer(dropout_out)
        dec_layer = keras.layers.Dense(input_shape, activation=activation, input_shape=[neurons_first_layer],
                                       kernel_initializer=init)
        dec_out = dec_layer(enc_out)

        model = keras.models.Model(inputs=input, outputs=dec_out)

        model.compile(loss=loss_function, optimizer=pretrain_optimizer, metrics=['accuracy'])

        hist = model.fit(x=self.x_train, y=self.x_train, epochs=pretrain_epochs, verbose=verbose,
                         validation_data=[self.x_valid, self.x_valid])

        self.layers_history.append(hist)

        decoder_layer_list = [dec_layer]

        enc_out = enc_layer(input)
        trained_encoder = keras.models.Model(inputs=input, outputs=enc_out)

        # add and train more layers
        full_model = self.add_and_fit_layers(loss_function, lr, pretrain_epochs, decoder_layer_list, neuron_list,
                                             dropout_rate, trained_encoder, input, verbose, pretrain_optimizer,
                                             activation, init)

        # finetune_model and return history

        print(f'Finetuning final model for {finetune_epochs} epochs')

        self.full_AE_train_history = full_model.fit(x=self.x_train, y=self.x_train, epochs=finetune_epochs,
                                                    batch_size=256, verbose=verbose,
                                                    validation_data=[self.x_valid, self.x_valid])

        # Saving full autoencoder because why not?
        self.autoencoder = keras.models.clone_model(full_model)
        self.autoencoder.set_weights(full_model.get_weights())
        self.autoencoder.compile(loss=loss_function, optimizer=pretrain_optimizer, metrics=['accuracy'])

        # extract encoder from autoencoder
        out = full_model.layers[1](input)
        full_encoder = keras.models.Model(inputs=input, outputs=out)
        full_encoder.compile(loss=loss_function, optimizer=keras.optimizers.SGD(lr), metrics=['accuracy'])

        self.encoder = full_encoder

    def add_and_fit_layers(self, loss_function, lr, pretrain_epochs, decoder_layer_list, neuron_list, dropout_rate,
                           current_encoder_stack, input, verbose, pretrain_optimizer, activation, init):

        which_layer = len(neuron_list)

        if which_layer == 0:
            return self.combine_encoder_decoder(current_encoder_stack, decoder_layer_list, input, loss_function,
                                                pretrain_optimizer)

        print(f'Adding and training another layer for {pretrain_epochs} epochs')

        # encode data using already trained encoders. Creates data used for training following layers
        encoded_data_train = current_encoder_stack.predict(self.x_train)
        #encoded_data_valid = current_encoder_stack.predict(self.x_valid)
        encoded_data_length = len(encoded_data_train[0])

        # Build new encoder + decoder pair
        n_neurons = neuron_list.pop(0)

        input_new_layer = keras.layers.Input(shape=(encoded_data_length,))

        dropout_out = keras.layers.Dropout(dropout_rate)(input_new_layer)

        # last encoder layer should not have activation function. allowing full expressiveness
        if which_layer == 1:
            new_enc_layer = keras.layers.Dense(n_neurons, input_shape=[encoded_data_length], kernel_initializer=init)
            new_enc_out = new_enc_layer(dropout_out)
        else:
            new_enc_layer = keras.layers.Dense(n_neurons, activation=activation, input_shape=[encoded_data_length],
                                               kernel_initializer=init)
            new_enc_out = new_enc_layer(dropout_out)

        # Add dropout before decoder layer
        dropout_decoder_out = keras.layers.Dropout(dropout_rate)(new_enc_out)

        # Add new decoder layer
        new_dec_layer = keras.layers.Dense(encoded_data_length, activation=activation, input_shape=[n_neurons],
                                           kernel_initializer=init)
        new_dec_out = new_dec_layer(dropout_decoder_out)

        # Compile + fit
        model = keras.models.Model(inputs=input_new_layer, outputs=new_dec_out)
        model.compile(loss=loss_function, optimizer=pretrain_optimizer, metrics=['accuracy'])
        hist = model.fit(x=encoded_data_train, y=encoded_data_train, epochs=pretrain_epochs, verbose=verbose,
                         validation_data=[self.x_valid, self.x_valid])

        self.layers_history.append(hist)

        # Store decoder layer
        # puts in the opposite end of append()
        decoder_layer_list.insert(0, new_dec_layer)

        # add new encoder layer to previous encoder layers
        old_out = current_encoder_stack(inputs=input)
        out = new_enc_layer(old_out)

        model = keras.models.Model(inputs=input, outputs=out)
        # skal man compile her?
        model.compile(loss=loss_function, optimizer=pretrain_optimizer, metrics=['accuracy'])

        return self.add_and_fit_layers(loss_function, lr, pretrain_epochs, decoder_layer_list, neuron_list,
                                       dropout_rate, model, input, verbose, pretrain_optimizer, activation, init)

    def combine_encoder_decoder(self, encoder, decoder_layers, input, loss_function, optimizer):

        out = encoder(input)
        for layer in decoder_layers:
            out = layer(out)

        model = keras.models.Model(inputs=input, outputs=out)
        # skal man compile her?
        model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

        return model

    def sådan_virker_pretraining(self, loss_function=keras.losses.binary_crossentropy, lr=1, number_of_epochs=10,
                                 input_shape=136):

        input = keras.layers.Input(shape=(input_shape,))

        enc_layer = keras.layers.Dense(500, activation='selu', input_shape=[input_shape],
                                       kernel_initializer=keras.initializers.lecun_normal())
        enc_out = enc_layer(input)
        dec_layer = keras.layers.Dense(input_shape, activation='selu', input_shape=[500],
                                       kernel_initializer=keras.initializers.lecun_normal())
        dec_out = dec_layer(enc_out)

        model = keras.models.Model(inputs=input, outputs=dec_out)

        model.compile(loss=loss_function, optimizer=keras.optimizers.SGD(lr), metrics=['accuracy'])

        model.fit(x=self.x_train, y=self.x_train, epochs=number_of_epochs, validation_data=[self.x_valid, self.x_valid])

        model_clone = keras.models.clone_model(model)
        model_clone.set_weights(model.get_weights())

        enc_layer.trainable = False

        enc2 = enc_layer(input)
        dec2_out = dec_layer(enc2)

        model2 = keras.models.Model(inputs=input, outputs=dec2_out)

        model2.compile(loss=loss_function, optimizer=keras.optimizers.SGD(lr), metrics=['accuracy'])

        model2.fit(x=self.x_train, y=self.x_train, epochs=number_of_epochs,
                   validation_data=[self.x_valid, self.x_valid])

        '''
        for x in range(1,3):
            m1 = model.layers[x].get_weights()
           m2 = model2.layers[x].get_weights()

           print(np.array_equal(m1[0], m2[0]))
           print(np.array_equal(m1[1], m2[1]))
        '''

        for x in range(1, 3):
            m1 = model_clone.layers[x].get_weights()
            m2 = model2.layers[x].get_weights()

            print(np.array_equal(m1[0], m2[0]))
            print(np.array_equal(m1[1], m2[1]))

        print("Sæt breakpoint på mig")


class DEC_Binner_Xifeng(DEC):
    def __init__(self, split_value, contig_ids, clustering_method, feature_matrix, log_dir):
        super().__init__(split_value=split_value, contig_ids=contig_ids, feature_matrix=feature_matrix,
                         clustering_method=clustering_method, log_dir=log_dir)
        self._input_layer_size = None
        self.log_dir = f'{self.log_dir}/DEC_XIFENG'

    def do_binning(self, init='glorot_uniform', pretrain_optimizer=keras.optimizers.Adam(learning_rate=0.0001), n_clusters=10, update_interval=140,
                   pretrain_epochs=10, batch_size=128, tolerance_threshold=1e-3,
                   max_iterations=100, true_bins=None):
        self.n_clusters = n_clusters
        self.autoencoder, self.encoder = self.define_model(dims=[self.x_train.shape[-1], 100, 100, 50, 10],
                                                           act='relu', init=init)

        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output)
        self.model = keras.models.Model(inputs=self.encoder.input, outputs=clustering_layer)

        self.pretrain(x=self.x_train, optimizer=pretrain_optimizer,
                      epochs=pretrain_epochs, batch_size=batch_size)

        self.model.summary()
        t0 = time()
        self.compile(optimizer=keras.optimizers.SGD(0.01, 0.9), loss='kld')
        y_pred, loss_list = self.fit(x=self.feature_matrix, y=None, tolerance_threshold=tolerance_threshold,
                                     maxiter=max_iterations, batch_size=batch_size, update_interval=update_interval)

        print('clustering time: ', (time() - t0))
        self.bins = y_pred
        self.cluster_loss_list = loss_list
        latent_representation = self.encoder.predict(self.feature_matrix)
        np.save('latent_representation',latent_representation)
    def pretrain(self, x, optimizer='adam', epochs=200, batch_size=256):
        print('...Pretraining...')

        self.autoencoder.compile(optimizer=optimizer, loss=self.custom_vamb_loss(0.05))

        log_dir = f'{self.log_dir}_pretrain_{int(self.train_start)}'

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # begin pretraining
        t0 = time()
        self.full_AE_train_history = self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs,
                                                          validation_data=[self.x_valid, self.x_valid],
                                                          callbacks=[tensorboard_callback])
        print('Pretraining time: %ds' % round(time() - t0))

        # TODO Save weights of pretrained model
        # self.autoencoder.save_weights(save_dir + '/ae_weights.h5')
        # print('Pretrained weights are saved to %s/ae_weights.h5' % save_dir)

    def extract_features(self, x):
        return self.encoder.predict(x)

    def compile(self, optimizer='sgd', loss='kld'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def define_model(self, dims, act='relu', init='glorot_uniform'):
        """
            Fully connected auto-encoder model, symmetric.
            Arguments:
                dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
                    The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
                act: activation, not applied to Input, Hidden and Output layers
            return:
                (ae_model, encoder_model), Model of autoencoder and model of encoder
            """

        n_stacks = len(dims) - 1
        # input
        x = keras.layers.Input(shape=(dims[0],), name='input')
        h = x

        # internal layers in encoder
        for i in range(n_stacks - 1):
            h = keras.layers.Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)

        # hidden layer
        h = keras.layers.Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(
            h)  # hidden layer, features are extracted from here

        y = h
        # internal layers in decoder
        for i in range(n_stacks - 1, 0, -1):
            y = keras.layers.Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)

        # output
        y = keras.layers.Dense(dims[0], kernel_initializer=init, name='decoder_0')(y)

        return keras.models.Model(inputs=x, outputs=y, name='AE'), keras.models.Model(inputs=x, outputs=h,
                                                                                      name='encoder')



def create_binner(split_value, binner_type, clustering_method, contig_ids, feature_matrix, log_dir, labels=None, x_train=None, x_valid=None ):
    clustering_method = get_clustering(clustering_method)

    binner_type = get_binner(binner_type)

    binner_instance = binner_type(split_value=split_value, contig_ids=contig_ids, clustering_method=clustering_method(),
                                  feature_matrix=feature_matrix, log_dir=log_dir, labels=labels, x_train=x_train, x_valid=x_valid)
    return binner_instance


def get_binner(binner):
    return binner_dict[binner]


def get_clustering(cluster):
    return clustering_algorithms_dict[cluster]


binner_dict = {
    'DEC': Greedy_pretraining_DEC,
    'SEQ': Sequential_Binner1,
    'DEC_XIFENG': DEC_Binner_Xifeng,
    'BAD': Badass_Binner
}

clustering_algorithms_dict = {
    'KMeans': clustering_methods.clustering_k_means,
    'Random': clustering_methods.random_cluster,
    'KMeans_gpu': clustering_methods.KMEANS_GPU,
    'DBSCAN_gpu': clustering_methods.DBSCAN_GPU
}
