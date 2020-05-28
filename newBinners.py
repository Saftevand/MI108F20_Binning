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
import matplotlib.ticker as ticker
import io
from collections import Counter
from tensorboard.plugins import projector
import matplotlib.ticker as ticker

class Binner(abc.ABC):
    def __init__(self, name, contig_ids, feature_matrix=None, labels=None, x_train=None, x_valid=None,
                 train_labels=None, valid_labels=None, pretraining_params=None, clust_params=None):
        self.input_shape = x_train.shape[1]
        self.training_set_size = x_train.shape[0]
        self.validation_set_size = x_valid.shape[0]
        self.feature_matrix = tf.constant(feature_matrix)
        self.x_train = tf.constant(x_train)
        self.x_valid = tf.constant(x_valid)
        self.contig_ids = contig_ids
        self.name = name

        self.pretraining_params = pretraining_params
        self.clust_params = clust_params


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

    def pretraining(self, callbacks):

        file_writer = tf.summary.create_file_writer(self.log_dir)

        train_loss = []
        x_train = tf.data.Dataset.from_tensor_slices(self.x_train)
        train_labels = tf.data.Dataset.from_tensor_slices(self.train_labels)
        validation_loss = 0
        loss_function = tf.keras.losses.get(self.autoencoder.loss)
        trained_samples = 0
        current_epoch = 0

        for epochs_to_run, batch_size in zip(self.pretraining_params['epochs'], self.pretraining_params['batch_sizes']):
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
                for callback in callbacks:
                    callback.on_epoch_end(current_epoch + 1)
                current_epoch += 1
        for callback in callbacks:
            callback.on_train_end()

    def include_clustering_loss(self):
        output_of_input = self.autoencoder.layers[0].output
        decoder_output = self.autoencoder.layers[-1].output
        latent_output = self.autoencoder.get_layer('latent_layer').output

        rerouted_autoencoder = keras.Model(inputs=[output_of_input], outputs=[decoder_output, latent_output],
                                           name='2outs_autoencoder')

        opt = keras.optimizers.Adam(learning_rate=self.clust_params['learning_rate'])

        rerouted_autoencoder.compile(optimizer=opt, loss=[self.clust_params['reconst_loss'], self.clust_params['clust_loss']],
                                     loss_weights=self.clust_params['loss_weights'])

        print(rerouted_autoencoder.summary())
        return rerouted_autoencoder

    def final_DBSCAN_clustering(self):
        dbscan_instance = sklearn.cluster.DBSCAN(eps=self.clust_params['eps'], min_samples=self.clust_params['min_samples'])

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
                 x_valid=None, train_labels=None,validation_labels=None, pretraining_params=None, clust_params=None):
        super().__init__(name=name, contig_ids=contig_ids, feature_matrix=feature_matrix, labels=labels,
                         x_train=x_train, x_valid=x_valid, train_labels=train_labels, valid_labels=validation_labels,
                         pretraining_params=pretraining_params, clust_params=clust_params)

    def create_stacked_AE(self):
        # get params from param dict
        input_shape = self.input_shape
        lr = self.pretraining_params['learning_rate']
        reconst_loss = self.pretraining_params['reconst_loss']
        activation_fn = self.pretraining_params['activation_fn']
        regularizer = self.pretraining_params['regularizer']
        init_fn = self.pretraining_params['initializer']
        no_neurons_hidden = self.pretraining_params['layer_size']
        num_hidden_layers = self.pretraining_params['num_hidden_layers']
        no_neurons_embedding = self.pretraining_params['emnedding_neurons']
        drop = self.pretraining_params['dropout']
        drop_rate = self.pretraining_params['drop_and_denoise_rate']
        denoise = self.pretraining_params['denoise']
        bn = self.pretraining_params['BN']

        if drop and denoise:
            print("HOLD STOP. DROPOUT + DENOISING det er for meget")
            return

        # Create input layer
        encoder_input = tf.keras.layers.Input(shape=(input_shape,), name="input")
        layer = encoder_input

        layer = tf.keras.layers.Dropout(drop_rate)(layer) if denoise else layer

        # Create encoding layers
        for layer_index in range(num_hidden_layers):
            layer = tf.keras.layers.Dropout(drop_rate)(layer) if drop else layer
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

        for layer_index in range(num_hidden_layers):
            layer = tf.keras.layers.Dropout(drop_rate)(layer) if drop else layer
            layer = tf.keras.layers.BatchNormalization()(layer) if bn else layer
            layer = tf.keras.layers.Dense(units=no_neurons_hidden, activation=activation_fn, kernel_initializer=init_fn,
                                          kernel_regularizer=regularizer,
                                          name=f'decoding_layer_{layer_index}')(layer)

        layer = tf.keras.layers.Dropout(drop_rate)(layer) if drop else layer
        layer = tf.keras.layers.BatchNormalization()(layer) if bn else layer
        decoder_output = tf.keras.layers.Dense(units=input_shape, kernel_initializer=init_fn,
                                               name='Decoder_output_layer')(layer)

        stacked_ae = tf.keras.Model(encoder_input, decoder_output, name='autoencoder')

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        stacked_ae.compile(optimizer=optimizer, loss=reconst_loss)

        print(stacked_ae.summary())

        return stacked_ae

    def do_binning(self, load_model=False, load_clustering_AE = True):
        '''Method is ALMOST identical to Sparse_AE... uses Stacked AE instead'''
        if load_model:
            self.autoencoder = tf.keras.models.load_model('autoencoder.h5')
        else:
            self.autoencoder = self.create_stacked_AE()
            callback_projector = ProjectEmbeddingCallback(binner=self)
            callback_activity = ActivityCallback(binner=self)

            self.pretraining(callbacks=[callback_projector, callback_activity])
            self.autoencoder.save('autoencoder.h5')
            print("AE saved")

        self.encoder = self.extract_encoder()
        self.encoder.save('encoder.h5')
        print("Encoder saved")

        if load_clustering_AE:
            self.autoencoder = tf.keras.models.load_model("clustering_AE.h5")
            self.encoder = self.extract_encoder()
        else:
            self.autoencoder = self.include_clustering_loss()
            self.fit_dbscan()
            self.autoencoder.save('clustering_autoencoder.h5')
            print("clustering_AE saved")

        self.bins = self.final_DBSCAN_clustering()
        return self.bins

    def fit_dbscan(self):
        x = self.feature_matrix
        y = self.labels
        batch_size = self.clust_params['batch_size']
        epochs = self.clust_params['epochs']
        eps = self.clust_params['eps']
        min_samples = self.clust_params['min_samples']

        callback_projector = ProjectEmbeddingCallback(binner=self, log_dir=os.path.join(self.log_dir, 'DeepClustering'))
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


class Sparse_Binner(Stacked_Binner):

    def create_sparse_AE(self):
        input_shape = self.input_shape
        lr = self.pretraining_params['learning_rate']
        reconst_loss = self.pretraining_params['reconst_loss']
        activation_fn = self.pretraining_params['activation_fn']
        regularizer = self.pretraining_params['regularizer']
        init_fn = self.pretraining_params['initializer']
        no_neurons_hidden = self.pretraining_params['layer_size']
        num_hidden_layers = self.pretraining_params['num_hidden_layers']
        no_neurons_embedding = self.pretraining_params['emnedding_neurons']
        drop = self.pretraining_params['dropout']
        drop_rate = self.pretraining_params['drop_and_denoise_rate']
        denoise = self.pretraining_params['denoise']
        bn = self.pretraining_params['BN']
        KLweight = self.pretraining_params['sparseKLweight']
        KLtarget = self.pretraining_params['sparseKLtarget']


        if drop and denoise:
            print("HOLD STOP. DROPOUT + DENOISING det er for meget")
            return

        # Create input layer
        encoder_input = tf.keras.layers.Input(shape=(input_shape,), name="input")
        layer = encoder_input

        layer = tf.keras.layers.Dropout(drop_rate)(layer) if denoise else layer

        # Create encoding layers
        for layer_index in range(num_hidden_layers):
            layer = tf.keras.layers.Dropout(drop_rate)(layer) if drop else layer
            layer = tf.keras.layers.BatchNormalization()(layer) if bn else layer
            layer = tf.keras.layers.Dense(units=no_neurons_hidden, activation=activation_fn, kernel_initializer=init_fn,
                                          kernel_regularizer=regularizer,
                                          name=f'encoding_layer_{layer_index}')(layer)

        #create regularizer
        KLDreg = KLDivergenceRegularizer(weight=KLweight, target=KLtarget)

        # create embedding layer
        layer = tf.keras.layers.BatchNormalization()(layer) if bn else layer
        embedding_layer = tf.keras.layers.Dense(units=no_neurons_embedding, kernel_initializer=init_fn, activation=tf.keras.activations.sigmoid, activity_regularizer=KLDreg,
                                                name='latent_layer')(layer)

        # Create decoding layers
        layer = embedding_layer

        for layer_index in range(num_hidden_layers):
            layer = tf.keras.layers.Dropout(drop_rate)(layer) if drop else layer
            layer = tf.keras.layers.BatchNormalization()(layer) if bn else layer
            layer = tf.keras.layers.Dense(units=no_neurons_hidden, activation=activation_fn, kernel_initializer=init_fn,
                                          kernel_regularizer=regularizer,
                                          name=f'decoding_layer_{layer_index}')(layer)

        layer = tf.keras.layers.Dropout(drop_rate)(layer) if drop else layer
        layer = tf.keras.layers.BatchNormalization()(layer) if bn else layer
        decoder_output = tf.keras.layers.Dense(units=input_shape, kernel_initializer=init_fn,
                                               name='Decoder_output_layer')(layer)

        stacked_ae = tf.keras.Model(encoder_input, decoder_output, name='autoencoder')

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        stacked_ae.compile(optimizer=optimizer, loss=reconst_loss)

        print(stacked_ae.summary())

        return stacked_ae

    def do_binning(self, load_model=False, load_clustering_AE=True):
        '''Method is ALMOST identical to Stacked_AE... uses sparse AE instead'''
        if load_model:
            self.autoencoder = tf.keras.models.load_model('autoencoder.h5', custom_objects={"KLDivergenceRegularizer": KLDivergenceRegularizer})
        else:
            self.autoencoder = self.create_sparse_AE()
            callback_projector = ProjectEmbeddingCallback(binner=self)
            callback_activity = ActivityCallback(binner=self)

            self.pretraining(callbacks=[callback_projector, callback_activity])
            self.autoencoder.save('autoencoder.h5')
            print("AE saved")

        self.encoder = self.extract_encoder()
        self.encoder.save('encoder.h5')
        print("Encoder saved")

        if load_clustering_AE:
            self.autoencoder = tf.keras.models.load_model("clustering_AE.h5", custom_objects={"KLDivergenceRegularizer": KLDivergenceRegularizer})
            self.encoder = self.extract_encoder()
        else:
            self.autoencoder = self.include_clustering_loss()
            self.fit_dbscan()
            self.autoencoder.save('clustering_AE.h5')
            print("clustering_AE saved")

        self.bins = self.final_DBSCAN_clustering()
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


def create_binner(binner_type, contig_ids, feature_matrix, labels=None, x_train=None, x_valid=None, train_labels=None,
                  validation_labels=None, pretraining_params=None, clust_params=None):
    binner = binner_dict[binner_type]

    binner_instance = binner(name=binner_type, contig_ids=contig_ids, feature_matrix=feature_matrix, labels=labels,
                                  x_train=x_train, x_valid=x_valid,train_labels=train_labels, validation_labels=validation_labels, pretraining_params=pretraining_params, clust_params=clust_params)
    return binner_instance



class ProjectEmbeddingCallback(tf.keras.callbacks.Callback):

    def __init__(self, binner, log_dir=None):
        super().__init__()
        self.binner = binner
        if log_dir == None:
            self.log_dir = self.binner.log_dir
        else:
            self.log_dir = log_dir
        self.embeddings = []
        self.config = projector.ProjectorConfig()

    def on_epoch_end(self, epoch, logs=None):
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
            pca = sklearn.decomposition.PCA(n_components=number_components)
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

    def __init__(self, binner):
        self.binner = binner
        self.file_writer = tf.summary.create_file_writer(binner.log_dir)
        super().__init__()

    def on_train_end(self):
        pass

    def on_epoch_end(self, epoch, logs=None):
        encoder = self.binner.extract_encoder()
        activations = encoder.predict(self.binner.x_train)
        activations_flatten = activations.flatten()
        #activations_percentages = activations.flatten()/activations.size
        number_of_activations = activations.size

        activation_percentage_figure, ax = plt.subplots()
        y_vals, x_vals, e_ = ax.hist(activations_flatten, edgecolor='black')
        ax.set_title("Embedding activations")
        ax.set_xlabel("Activations")
        ax.set_ylabel("% Activation")

        y_max = round((max(y_vals) / number_of_activations) + 0.01, 2)
        y_max = 1
        ax.set_yticks(ticks=np.arange(0.0, y_max * number_of_activations, 0.1 * number_of_activations,))
        ax.set_ylim(ax.get_yticks()[0], ax.get_yticks()[-1])
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=number_of_activations))
        mean_activities = tf.reduce_mean(activations, axis=0).numpy()
        number_neurons = mean_activities.size
        neurons_activity_figure, ax = plt.subplots()
        y_vals, x_vals, e_ = ax.hist(mean_activities, edgecolor='black')
        ax.set_title("Activity neuron")
        ax.set_xlabel("Neuron Mean Activation")
        ax.set_ylabel("% Neurons")
        y_max = round((max(y_vals) / number_neurons) + 0.01, 2)
        ax.set_yticks(ticks=np.arange(0.0, y_max * number_neurons, 0.05 * number_neurons, ))
        ax.set_ylim(ax.get_yticks()[0], ax.get_yticks()[-1])
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=number_neurons))
        with self.file_writer.as_default():
            tf.summary.image("Activation histogram",plot_to_image(activation_percentage_figure), step=epoch)
            tf.summary.image("Neuron histogram", plot_to_image(neurons_activity_figure), step=epoch)





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

binner_dict = {
    'STACKED': Stacked_Binner,
    'SPARSE': Sparse_Binner
}