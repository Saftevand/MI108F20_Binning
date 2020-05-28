import abc
import sys
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
import os
import sklearn
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from collections import Counter
from tensorboard.plugins import projector

class Binner(abc.ABC):
    def __init__(self, name, contig_ids, feature_matrix=None, labels=None, x_train=None, x_valid=None, train_labels=None, valid_labels=None):
        self.input_shape = x_train.shape[1]
        self.training_set_size = x_train.shape[0]
        self.validation_set_size = x_valid.shape[0]
        self.feature_matrix = tf.constant(feature_matrix)
        self.x_train = tf.constant(x_train)
        self.x_valid = tf.constant(x_valid)
        self.contig_ids = contig_ids
        self.name = name


        if labels is not None:
            self.labels = tf.constant(labels)
            self.train_labels = tf.constant(train_labels)
            self.validation_labels = tf.constant(valid_labels)
        self.autoencoder = None
        self.encoder = None
        self.clustering_autoencoder = None

        self.log_dir = os.path.join('Logs', f'{time.strftime("run_%Y_%m_%d-%H_%M_%S")}_{self.name}')
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
        #print(encoder.summary())
        return encoder


    def pretraining(self, epochs, batch_sizes):

        callback_projector = ProjectEmbeddingCallback(binner=self)
        callback_projector.model = self
        file_writer = tf.summary.create_file_writer(self.log_dir)

        train_loss = []
        x_train = tf.data.Dataset.from_tensor_slices(self.x_train)
        train_labels = tf.data.Dataset.from_tensor_slices(self.train_labels)
        validation_loss = 0
        loss_function = tf.keras.losses.get(self.autoencoder.loss)
        trained_samples = 0
        current_epoch = 0

        for epochs_to_run, batch_size in zip(epochs, batch_sizes):
            training_set = x_train.shuffle(buffer_size=40000, seed=2).repeat().batch(batch_size)
            training_labels = train_labels.shuffle(buffer_size=40000, seed=2).repeat().batch(batch_size)

            for epoch in range(epochs_to_run):
                for batch in training_set:
                    trained_samples += batch_size
                    train_loss.append(self.autoencoder.train_on_batch(batch, batch))
                    if trained_samples >= self.training_set_size:
                        trained_samples = 0
                        break

                # Epoch over - get metrics

                reconstruction = self.autoencoder(self.x_valid, training=False)
                validation_loss = np.mean(loss_function(reconstruction, self.x_valid))
                mean_train_loss = np.mean(train_loss)
                print(
                    f'Epoch\t{current_epoch + 1}\t\tTraining_loss:{mean_train_loss:.4f}\tValidation_loss:{validation_loss:.4f}')
                with file_writer.as_default():
                    tf.summary.scalar('Training loss', mean_train_loss, step=current_epoch + 1)
                    tf.summary.scalar('Validation loss', validation_loss, step=current_epoch + 1)
                train_loss.clear()

                callback_projector.on_epoch_end(current_epoch + 1)
                current_epoch += 1
        callback_projector.on_train_end()


    def include_clustering_loss(self, learning_rate=0.001, loss_weights=[0.5, 0.5]):
        output_of_input = self.autoencoder.layers[0].output
        decoder_output = self.autoencoder.layers[-1].output
        latent_output = self.autoencoder.get_layer('latent_layer').output

        rerouted_autoencoder = keras.Model(inputs=[output_of_input], outputs=[decoder_output, latent_output],
                                           name='2outs_autoencoder')

        opt = keras.optimizers.Adam(learning_rate=learning_rate)

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

    def __init__(self, name, contig_ids, feature_matrix, labels=None, x_train=None,
                 x_valid=None, train_labels=None,validation_labels=None):
        super().__init__(name=name, contig_ids=contig_ids, feature_matrix=feature_matrix, labels=labels, x_train=x_train, x_valid=x_valid, train_labels=train_labels, valid_labels=validation_labels)



    def create_stacked_AE(self, x_train, x_valid, no_layers=3, no_neurons_hidden=200,
                                                       no_neurons_embedding=32, epochs=100, drop=False, bn=False,
                                                       denoise=False, regularizer=None, lr=0.001):
        if drop and denoise:
            print("HOLD STOP. DROPOUT + DENOISING det er for meget")
            return

        # split data
        input_shape = self.input_shape
        activation_fn = 'elu'
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
            self.pretraining(epochs=[12,23,34], batch_sizes=[32, 64, 128])
            self.autoencoder.save('autoencoder.h5')
            print("AE saved")

        self.encoder = self.extract_encoder()
        self.encoder.save('encoder.h5')
        print("Encoder saved")

        #DBSCAN params
        eps=0.5
        min_samples=10
        if load_clustering_AE:
            self.autoencoder = tf.keras.models.load_model("clustering_AE.h5")
            self.encoder = self.extract_encoder()
        else:
            self.autoencoder = self.include_clustering_loss(learning_rate=0.0001, loss_weights=[1, 0.05])
            self.fit_dbscan(self.feature_matrix, y=self.labels, batch_size=4000, epochs=20, cuda=True, eps=eps, min_samples=min_samples)
            self.autoencoder.save('clustering_autoencoder.h5')
            print("clustering_AE saved")

        self.bins = self.final_DBSCAN_clustering(eps=eps, min_samples=min_samples)
        return self.bins


    def fit_dbscan(self, x, y=None, batch_size=4000, epochs=10, cuda=True, eps=0.5, min_samples=10):

        callback_projector = ProjectEmbeddingCallback(binner=self, log_dir=os.path.join(self.log_dir,'DeepClustering'))
        callback_projector.model = self
        file_writer = tf.summary.create_file_writer(self.log_dir)

        reconstruction_loss = []
        clustering_loss = []
        combined_loss = []

        tensorboard = keras.callbacks.TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=0,
            batch_size=batch_size,
            write_graph=True,
            write_grads=True
        )

        data = np.copy(x)
        labels = y.numpy()
        indices = np.arange(len(data))

        no_batches = len(x) / batch_size


        tensorboard.set_model(self.autoencoder)

        for i in range(epochs):

            np.random.RandomState(i).shuffle(data)
            np.random.RandomState(i).shuffle(indices)
            np.random.RandomState(i).shuffle(labels)
            batches = np.array_split(data, no_batches)
            batch_indices_list = np.array_split(indices, no_batches)
            amount_batches = len(batches)

            print(f'Current epoch: {i + 1} of total epochs: {epochs}')

            for batch_no, batch in enumerate(batches):
                batch_indices = batch_indices_list[batch_no]
                print(f'Current batch: {batch_no + 1} of total batches: {amount_batches}')

                # 2. encode all data
                encoded_data_full = self.encoder.predict(data)

                # 3. cluster

                current_time = time.strftime('%H:%M:%S')
                print(current_time)

                dbscan_instance = DBSCAN(eps=eps, min_samples=min_samples)
                dbscan_instance.fit(encoded_data_full)

                # Todo this should be np array
                all_assignments = np.array(dbscan_instance.labels_)


                batch_assignments = np.empty(len(batch_indices))
                for count, q in enumerate(batch_indices):
                    batch_assignments[count] = all_assignments[q]

                batch_centroids = []

                centroid_dict = {}

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

                list_of_losses = self.autoencoder.train_on_batch(x=[batch], y=[batch, np.array(batch_centroids)])

                combined_loss.append(list_of_losses[0])
                reconstruction_loss.append(list_of_losses[1])
                clustering_loss.append(list_of_losses[2])

            print(
                f'Epoch\t{i + 1}\t\tReconstruction_loss:{np.mean(reconstruction_loss):.4f}\tClustering_loss:{np.mean(clustering_loss):.4f}\tTotal_loss:{np.mean(combined_loss):.4f}')
            with file_writer.as_default():
                tf.summary.scalar('Reconstruction loss', np.mean(reconstruction_loss), step=i + 1)
                tf.summary.scalar('Clustering loss', np.mean(clustering_loss), step=i + 1)
                tf.summary.scalar('Total loss', np.mean(combined_loss), step=i + 1)
            reconstruction_loss.clear(), clustering_loss.clear(), combined_loss.clear()

            callback_projector.on_epoch_end(i + 1)

        callback_projector.on_train_end()
        tensorboard.on_train_end(None)

def print_status_bar(iteration, total, loss, metrics=None):
    metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result()) for m in [loss] + (metrics or [])])
    end = "" if iteration < total else "\n"
    print("\r{}/{} - ".format(iteration, total) + metrics, end=end)

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


def create_binner(binner_type, contig_ids, feature_matrix, labels=None, x_train=None, x_valid=None, train_labels=None,validation_labels=None ):
    binner = binner_dict[binner_type]

    binner_instance = binner(name=binner_type, contig_ids=contig_ids, feature_matrix=feature_matrix, labels=labels,
                                  x_train=x_train, x_valid=x_valid,train_labels=train_labels, validation_labels=validation_labels)
    return binner_instance


binner_dict = {
    'STACKED': Stacked_Binner,
    'SPARSE': Sparse_Binner
}

class ProjectEmbeddingCallback(tf.keras.callbacks.Callback):

    def __init__(self, binner, log_dir=None, project_every_modulo=10):
        super().__init__()
        self.binner = binner
        if log_dir == None:
            self.log_dir = self.binner.log_dir
        else:
            self.log_dir = log_dir
        self.project_every_modulo = project_every_modulo
        self.embeddings = []
        self.config = projector.ProjectorConfig()

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.project_every_modulo == 0:
            embeddings = self.binner.extract_encoder()(self.binner.feature_matrix)
            data_to_project = self.project_data(embedding=embeddings, epoch=epoch)
            data_variable = tf.Variable(data_to_project, name=f'epoch{epoch}')
            self.embeddings.append(data_variable)
    def on_train_end(self, logs=None):
        if len(self.embeddings) != 0:
            saver = tf.compat.v1.train.Saver(var_list=self.embeddings, save_relative_paths=True)
            saver.save(None, os.path.join(self.log_dir, 'embeddings.ckpt'))
            for e in self.embeddings:
                embedding = self.config.embeddings.add()
                embedding.tensor_name = e.name
                embedding.metadata_path = "metadata.tsv"
                projector.visualize_embeddings(self.log_dir, self.config)



    def project_data(self, embedding, bins_to_plot=10, run_on_all_data=False,
                     number_components=10, epoch=-1):

        number_of_contigs = embedding.shape[0]
        labels = self.binner.labels.numpy()
        counter = Counter(labels)
        # bins_atleast_10_contigs = [key for key, val in counter.items() if val >= 10]

        sorted_by_count = sorted(list(counter.items()), key=lambda x: x[1], reverse=True)
        samples_of_bins = [fst for fst, snd in sorted_by_count[1:11]]
        # samples_of_bins = random.choices(bins_atleast_10_contigs, k=bins_to_plot)

        if run_on_all_data:
            pca = decomposition.PCA(n_components=number_components)
            pca.fit(embedding)
            data_to_project = pca.transform(embedding)
            plt.scatter(data_to_project[:, 0], data_to_project[:, 1])

            with open(os.path.join(self.binner.log_dir, 'metadata.tsv'), "w") as f:
                for _, bin_id in self.assignments.items():
                    if bin_id in samples_of_bins:
                        f.write(f'{bin_id}\n')
                    else:
                        f.write(f'-1\n')
        else:
            mask = np.zeros(number_of_contigs, dtype=bool)

            if os.path.exists(os.path.join(self.binner.log_dir, 'metadata.tsv')):
                for (contig_idx, bin_id) in zip(range(number_of_contigs), labels):
                    if bin_id in samples_of_bins:
                        mask[contig_idx] = True
            # with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
            else:
                with open(os.path.join(self.binner.log_dir, 'metadata.tsv'), "w") as f:
                    for (contig_idx, bin_id) in zip(range(number_of_contigs), labels):
                        if bin_id in samples_of_bins:
                            mask[contig_idx] = True
                            f.write(f'{bin_id}\n')

            data_to_project = embedding[mask]
            #pca = decomposition.PCA(n_components=number_components)
            #pca.fit(data_to_project)
            #data_to_project = pca.transform(data_to_project)
            #sns.scatterplot(data_to_project[:, 0], data_to_project[:, 1])
            #plt.show()

        #variance_of_components = pca.explained_variance_ratio_

        return data_to_project


class ActivityCallback(tf.keras.callbacks.Callback):

    def __init__(self, training_data):
        self.training_data = training_data
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        encoder = self.model.get_layer('Encoder')
        activations = encoder.predict(self.training_data)
        activations_flatten = activations.flatten()
        #activations_percentages = activations.flatten()/activations.size
        number_of_activations = activations.size
        fig, ax = plt.subplots()
        y_vals, x_vals, e_ = ax.hist(activations_flatten, edgecolor='black')
        ax.set_title("Embedding activations")
        ax.set_xlabel("Activations")
        ax.set_ylabel("% Activation")

        y_max = round((max(y_vals) / number_of_activations) + 0.01, 2)
        y_max = 1
        ax.set_yticks(ticks=np.arange(0.0, y_max * number_of_activations, 0.1 * number_of_activations,))
        ax.set_ylim(ax.get_yticks()[0], ax.get_yticks()[-1])
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=number_of_activations))
        plt.show()

        mean_activities = tf.reduce_mean(activations, axis=0).numpy()
        number_neurons = mean_activities.size
        fig, ax = plt.subplots()
        y_vals, x_vals, e_ = ax.hist(mean_activities, edgecolor='black')
        ax.set_title("Activity neuron")
        ax.set_xlabel("Neuron Mean Activation")
        ax.set_ylabel("% Neurons")
        y_max = round((max(y_vals) / number_neurons) + 0.01, 2)
        ax.set_yticks(ticks=np.arange(0.0, y_max * number_neurons, 0.05 * number_neurons, ))
        ax.set_ylim(ax.get_yticks()[0], ax.get_yticks()[-1])
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=number_neurons))
        plt.show()
