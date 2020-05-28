import abc
import sys
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
import os
import sklearn
from collections import Counter

class Binner(abc.ABC):
    def __init__(self, contig_ids, feature_matrix=None, labels=None, x_train=None, x_valid=None):
        self.feature_matrix = feature_matrix
        self.x_train = x_train
        self.x_valid = x_valid
        self.contig_ids = contig_ids
        self.labels = labels

        self.autoencoder = None
        self.encoder = None
        self.clustering_autoencoder = None

        self.log_dir = os.path.join('Logs', time.strftime("run_%Y_%m_%d-%H_%M_%S"))

        self.bins = None


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

    def extract_encoder(self):
        input = self.autoencoder.get_layer('input').output
        latent_output = self.autoencoder.get_layer('latent_layer').output
        encoder = keras.Model(input, latent_output, name="encoder")
        print(encoder.summary())
        return encoder

    def extract_clustering_encoder(self):
        input = self.clustering_autoencoder.get_layer('input').output
        latent_output = self.clustering_autoencoder.get_layer('latent_layer').output
        encoder = keras.Model(input, latent_output, name="encoder")
        print(encoder.summary())
        return encoder

    def pretraining(self, stacked_ae, x_train, x_valid):

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)

        history = stacked_ae.fit(x_train, x_train, epochs=100, batch_size=32, validation_data=(x_valid, x_valid),
                                 shuffle=True, callbacks=[tensorboard_callback])
        history = stacked_ae.fit(x_train, x_train, epochs=100, batch_size=64, validation_data=(x_valid, x_valid),
                                 shuffle=True, callbacks=[tensorboard_callback])
        history = stacked_ae.fit(x_train, x_train, epochs=150, batch_size=128, validation_data=(x_valid, x_valid),
                                 shuffle=True, callbacks=[tensorboard_callback])
        history = stacked_ae.fit(x_train, x_train, epochs=150, batch_size=256, validation_data=(x_valid, x_valid),
                                 shuffle=True, callbacks=[tensorboard_callback])
        history = stacked_ae.fit(x_train, x_train, epochs=300, batch_size=512, validation_data=(x_valid, x_valid),
                                 shuffle=True, callbacks=[tensorboard_callback])
        history = stacked_ae.fit(x_train, x_train, epochs=300, batch_size=1024, validation_data=(x_valid, x_valid),
                                 shuffle=True, callbacks=[tensorboard_callback])

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

        print(rerouted_autoencoder.summary())
        return rerouted_autoencoder

    def final_DBSCAN_clustering(self, eps=0.5, min_samples=10):
        dbscan_instance = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples)

        print("Final clustering...")
        enc = self.encoder.predict(self.feature_matrix)
        dbscan_instance.fit(enc)


        '''herfra ændringer'''
        all_assignments = np.array(dbscan_instance.labels_)

        batch_centroids = []
        centroid_dict = {}
        final_assignments = all_assignments.copy()
        for cluster_label in set(all_assignments):
            # if outlier
            if cluster_label == -1:
                continue

            cluster_mask = (all_assignments == cluster_label)
            encoded_cluster = enc[cluster_mask]
            centroid = np.mean(encoded_cluster, axis=0)
            centroid_dict[cluster_label] = centroid

        centroids = np.vstack(list(centroid_dict.values()))

        for index, (contig, cluster_label) in enumerate(zip(enc, all_assignments)):
            if cluster_label == -1:
                dists = np.sqrt(np.sum(((centroids - contig) ** 2), axis=1))
                shortest_dist = np.argmin(dists)

                batch_centroids.append(centroid_dict[shortest_dist])
                final_assignments[index] = shortest_dist


        print("Binning complete.")

        print(f'final clusters {Counter(final_assignments)}')
        return final_assignments


class Stacked_Binner(Binner):

    def __init__(self, contig_ids, feature_matrix, labels=None, x_train=None,
                 x_valid=None):
        super().__init__(contig_ids=contig_ids, feature_matrix=feature_matrix, labels=labels, x_train=x_train, x_valid=x_valid)



    def create_stacked_AE(self, x_train, x_valid, no_layers=3, no_neurons_hidden=200,
                                                       no_neurons_embedding=32, epochs=100, drop=False, bn=False,
                                                       denoise=False, regularizer=None, lr=0.001):
        if drop and denoise:
            print("HOLD STOP. DROPOUT + DENOISING det er for meget")
            return

        # split data
        input_shape = x_train.shape[1]
        activation_fn = 'selu'
        init_fn = 'he_normal'


        # Create input layer
        encoder_input = tf.keras.layers.Input(shape=(input_shape,), name="input")
        layer = encoder_input

        if no_layers == 0:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
            init_fn = tf.keras.initializers.RandomNormal()

            layer = tf.keras.layers.Dense(units=no_neurons_embedding, activation=activation_fn,
                                          kernel_initializer=init_fn, kernel_regularizer=regularizer, name=f'Encoder')(
                layer)
            out = tf.keras.layers.Dense(units=input_shape, kernel_initializer=init_fn, name='Decoder_output_layer')(
                layer)

            small_AE = tf.keras.Model(encoder_input, out, name='Decoder')
            small_AE.compile(optimizer=optimizer, loss='mae')
            small_AE.fit(x_train, x_train, epochs=epochs, validation_data=(x_valid, x_valid), shuffle=True)
            return small_AE

        layer = tf.keras.layers.Dropout(.2)(layer) if denoise else layer

        # Create encoding layers
        for layer_index in range(no_layers):
            layer = tf.keras.layers.Dropout(.2)(layer) if drop else layer
            layer = tf.keras.layers.BatchNormalization()(layer) if bn else layer
            layer = tf.keras.layers.Dense(units=no_neurons_hidden, activation=activation_fn, kernel_initializer=init_fn,
                                          kernel_regularizer=regularizer,
                                          name=f'encoding_layer_{layer_index}')(layer)

        # create embedding layer
        layer = tf.keras.layers.BatchNormalization()(layer) if bn else layer
        embedding_layer = tf.keras.layers.Dense(units=no_neurons_embedding, kernel_initializer=init_fn,
                                                name='latent_layer')(layer)

        # Create decoding layers
        layer = embedding_layer

        for layer_index in range(no_layers):
            layer = tf.keras.layers.Dropout(.2)(layer) if drop else layer
            layer = tf.keras.layers.BatchNormalization()(layer) if bn else layer
            layer = tf.keras.layers.Dense(units=no_neurons_hidden, activation=activation_fn, kernel_initializer=init_fn,
                                          kernel_regularizer=regularizer,
                                          name=f'decoding_layer_{layer_index}')(layer)

        layer = tf.keras.layers.Dropout(.2)(layer) if drop else layer
        layer = tf.keras.layers.BatchNormalization()(layer) if bn else layer
        decoder_output = tf.keras.layers.Dense(units=input_shape, kernel_initializer=init_fn,
                                               name='Decoder_output_layer')(layer)

        stacked_ae = tf.keras.Model(encoder_input, decoder_output, name='autoencoder')

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        stacked_ae.compile(optimizer=optimizer, loss='mae')

        print(stacked_ae.summary())

        return stacked_ae

    def do_binning(self, load_model=False, load_clustering_AE = True):
        '''Method is ALMOST identical to Sparse_AE... uses Stacked AE instead'''
        if load_model:
            self.autoencoder = tf.keras.models.load_model('autoencoder.h5')
        else:
            self.autoencoder = self.create_stacked_AE(self.x_train, self.x_valid, no_layers=3, no_neurons_hidden=200,
                                                       no_neurons_embedding=32, epochs=100, drop=False, bn=False,
                                                       denoise=False, regularizer=tf.keras.regularizers.l1(), lr=0.001)
            self.pretraining(self.autoencoder, self.x_train, self.x_valid)
            self.autoencoder.save('autoencoder.h5')
            print("AE saved")

        self.encoder = self.extract_encoder()
        self.encoder.save('encoder.h5')
        print("Encoder saved")

        #DBSCAN params
        eps=0.5
        min_samples=10
        if load_clustering_AE:
            self.clustering_autoencoder = tf.keras.models.load_model("clustering_AE.h5")
            self.encoder = self.extract_clustering_encoder()
        else:
            self.clustering_autoencoder = self.include_clustering_loss(learning_rate=0.0001, loss_weights=[1, 0.05])
            self.fit_dbscan(self.feature_matrix, y=self.labels, batch_size=4000, epochs=1, cuda=True, eps=eps, min_samples=min_samples)
            self.clustering_autoencoder.save('clustering_AE.h5')
            print("clustering_AE saved")

        self.bins = self.final_DBSCAN_clustering(eps=eps, min_samples=min_samples)
        return self.bins

    def fit_dbscan(self, x, y=None, batch_size=4000, epochs=10, cuda=True, eps=0.5, min_samples=10):

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
            np.random.RandomState(i).shuffle(labels)
            batches = np.array_split(data, no_batches)
            batch_indices_list = np.array_split(indices, no_batches)
            amount_batches = len(batches)

            print(f'Current epoch: {i + 1} of total epochs: {epochs}')
            total_loss = 0
            reconstruct_loss = 0
            cluster_loss = 0
            for batch_no, batch in enumerate(batches):
                batch_indices = batch_indices_list[batch_no]
                print(f'Current batch: {batch_no + 1} of total batches: {amount_batches}')

                # 2. encode all data
                encoded_data_full = self.encoder.predict(data)

                # 3. cluster
                assigned_medoids_per_contig = np.zeros(encoded_data_full.shape)
                assignments = np.zeros(len(encoded_data_full))
                current_time = time.strftime('%H:%M:%S')
                print(current_time)

                dbscan_instance = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples)

                # dbscan_instance.fit(encoded_data_full)
                dbscan_instance.fit(encoded_data_full)

                # Todo this should be np array
                all_assignments = np.array(dbscan_instance.labels_)

                # mask = np.zeros(len(encoded_data_full), dtype=bool)
                # mask[batch_indices_list] = True
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

                complete_time = time.strftime('%H:%M:%S')
                print(complete_time)

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

                # truth_medoids = tf.gather(params=assigned_medoids_per_contig, indices=batch_indices)

                list_of_losses = self.clustering_autoencoder.train_on_batch(x=[batch], y=[batch, np.array(batch_centroids)])
                # print(self.autoencoder.metrics_names)
                # print(list_of_losses)

                total_loss += list_of_losses[0]
                reconstruct_loss += list_of_losses[1]
                cluster_loss += list_of_losses[2]

            history_obj = list()
            tot_avg = total_loss / amount_batches
            history_obj.append(tot_avg)

            recon_avg = reconstruct_loss / amount_batches
            history_obj.append(recon_avg)

            clust_avg = cluster_loss / amount_batches
            history_obj.append(clust_avg)

            print(f'Total loss: {tot_avg}, \t Reconst loss: {recon_avg}, \t Clust loss: {clust_avg}')

            # self.tsne_embeddings(epoch=i+1,encoded_data=encoded_data_full, labels=labels, medoids=None, assignments=None)

            tensorboard.on_epoch_end(i + 1, named_logs(self.autoencoder, history_obj))
        # self.embeddings_projector(epoch=i+1, labels=labels, assignments=all_assignments, medoids=None, encodings=encoded_data_full)
        tensorboard.on_train_end(None)


class Sparse_Binner(Stacked_Binner):
    #TODO how to save regularizer
    def create_sparse_AE(self, x_train, x_valid, no_layers=3, no_neurons_hidden=200, no_neurons_embedding=32,
                         epochs=100, drop=False, bn=False, denoise=False, regularizer=None, lr=0.001):
        if drop and denoise:
            print("HOLD STOP. DROPOUT + DENOISING det er for meget")
            return

        # split data
        input_shape = x_train.shape[1]
        activation_fn = 'selu'
        init_fn = 'he_normal'


        # Create input layer
        encoder_input = tf.keras.layers.Input(shape=(input_shape,), name="input")
        layer = encoder_input

        if no_layers == 0:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            init_fn = tf.keras.initializers.RandomNormal()

            layer = tf.keras.layers.Dense(units=no_neurons_embedding, activation=activation_fn,
                                          kernel_initializer=init_fn, kernel_regularizer=regularizer, name=f'Encoder')(
                layer)
            out = tf.keras.layers.Dense(units=input_shape, kernel_initializer=init_fn, name='Decoder_output_layer')(
                layer)

            small_AE = tf.keras.Model(encoder_input, out, name='Decoder')
            small_AE.compile(optimizer=optimizer, loss='mae')
            small_AE.fit(x_train, x_train, epochs=epochs, validation_data=(x_valid, x_valid), shuffle=True)
            return small_AE

        layer = tf.keras.layers.Dropout(.2)(layer) if denoise else layer

        # Create encoding layers
        for layer_index in range(no_layers):
            layer = tf.keras.layers.Dropout(.2)(layer) if drop else layer
            layer = tf.keras.layers.BatchNormalization()(layer) if bn else layer
            layer = tf.keras.layers.Dense(units=no_neurons_hidden, activation=activation_fn, kernel_initializer=init_fn,
                                          kernel_regularizer=regularizer,
                                          name=f'encoding_layer_{layer_index}')(layer)

        #create regularizer
        KLDreg = KLDivergenceRegularizer(weight=0.5, target=0.1)

        # create embedding layer
        layer = tf.keras.layers.BatchNormalization()(layer) if bn else layer
        embedding_layer = tf.keras.layers.Dense(units=no_neurons_embedding, kernel_initializer=init_fn, activation=tf.keras.activations.sigmoid, activity_regularizer=KLDreg,
                                                name='latent_layer')(layer)

        # Create decoding layers
        layer = embedding_layer

        for layer_index in range(no_layers):
            layer = tf.keras.layers.Dropout(.2)(layer) if drop else layer
            layer = tf.keras.layers.BatchNormalization()(layer) if bn else layer
            layer = tf.keras.layers.Dense(units=no_neurons_hidden, activation=activation_fn, kernel_initializer=init_fn,
                                          kernel_regularizer=regularizer,
                                          name=f'decoding_layer_{layer_index}')(layer)

        layer = tf.keras.layers.Dropout(.2)(layer) if drop else layer
        layer = tf.keras.layers.BatchNormalization()(layer) if bn else layer
        decoder_output = tf.keras.layers.Dense(units=input_shape, kernel_initializer=init_fn,
                                               name='Decoder_output_layer')(layer)

        stacked_ae = tf.keras.Model(encoder_input, decoder_output, name='autoencoder')

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        stacked_ae.compile(optimizer=optimizer, loss='mae')

        print(stacked_ae.summary())

        return stacked_ae

    def do_binning(self, load_model=False, load_clustering_AE=True):
        '''Method is ALMOST identical to Stacked_AE... uses sparse AE instead'''
        if load_model:
            self.autoencoder = tf.keras.models.load_model('autoencoder.h5', custom_objects={"KLDivergenceRegularizer": KLDivergenceRegularizer})
        else:
            self.autoencoder = self.create_sparse_AE(self.x_train, self.x_valid, no_layers=3, no_neurons_hidden=200,
                                                      no_neurons_embedding=32, epochs=100, drop=False, bn=False,
                                                      denoise=False, regularizer=None, lr=0.001)
            self.pretraining(self.autoencoder, self.x_train, self.x_valid)
            self.autoencoder.save('autoencoder.h5')
            print("AE saved")

        self.encoder = self.extract_encoder()
        self.encoder.save('encoder.h5')
        print("Encoder saved")

        # DBSCAN params
        eps = 0.5
        min_samples = 4
        if load_clustering_AE:
            self.clustering_autoencoder = tf.keras.models.load_model("clustering_AE.h5", custom_objects={"KLDivergenceRegularizer": KLDivergenceRegularizer})
            self.encoder = self.extract_clustering_encoder()
        else:
            self.clustering_autoencoder = self.include_clustering_loss(learning_rate=0.0001, loss_weights=[1, 0.05])
            self.fit_dbscan(self.feature_matrix, y=self.labels, batch_size=4000, epochs=1, cuda=True, eps=eps,
                            min_samples=min_samples)
            self.clustering_autoencoder.save('clustering_AE.h5')
            print("clustering_AE saved")

        self.bins = self.final_DBSCAN_clustering(eps=eps, min_samples=min_samples)
        return self.bins


class KLDivergenceRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, weight, target=0.1, **kwargs):
        self.weight = weight
        self.target = target
        super().__init__(**kwargs)

    def __call__(self, inputs):
        mean_activities = tf.reduce_mean(inputs, axis=0)
        return self.weight * (tf.keras.losses.kld(self.target, mean_activities) + tf.keras.losses.kld(1. - self.target,
                                                                                                      1. - mean_activities))

    def get_config(self):
        return {"weight" : self.weight, "target" : self.target}


def create_binner(binner_type, contig_ids, feature_matrix, labels=None, x_train=None, x_valid=None ):
    binner_type = binner_dict[binner_type]

    binner_instance = binner_type(contig_ids=contig_ids, feature_matrix=feature_matrix, labels=labels,
                                  x_train=x_train, x_valid=x_valid)
    return binner_instance


binner_dict = {
    'STACKED': Stacked_Binner,
    'SPARSE': Sparse_Binner
}
