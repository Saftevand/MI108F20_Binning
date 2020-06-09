import abc
import sys
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
import os
import sklearn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import io
from collections import Counter
from tensorboard.plugins import projector
import matplotlib.ticker as ticker
from sklearn.preprocessing import normalize
import hdbscan
import json
import data_processor
import seaborn as sns
import random


class Binner(abc.ABC):
    def __init__(self, name, contig_ids, feature_matrix=None, labels=None, x_train=None, x_valid=None,
                 train_labels=None, valid_labels=None, pretraining_params=None, clust_params=None, debug=False):
        self.input_shape = x_train.shape[1]
        self.training_set_size = x_train.shape[0]
        #self.validation_set_size = x_valid.shape[0]
        self.feature_matrix = tf.constant(feature_matrix)
        self.x_train = tf.constant(x_train)
        self.x_valid = tf.constant(x_valid)
        self.contig_ids = contig_ids
        self.name = name
        self.debug = debug
        self.pretraining_params = pretraining_params
        self.clust_params = clust_params


        if labels is not None:
            self.labels = tf.constant(labels)
            self.train_labels = tf.constant(train_labels)
            self.validation_labels = tf.constant(valid_labels)
        self.autoencoder = None
        self.encoder = None
        self.clustering_autoencoder = None

        # self.log_dir = os.path.join('Logs', f'{time.strftime("run_%Y_%m_%d-%H_%M_%S")}_{self.name}')
        # bedre formatering
        if self.pretraining_params["dropout"] or self.pretraining_params["denoise"]:
            dr = self.pretraining_params["drop_and_denoise_rate"]
        else:
            dr = 0
        self.formatted = f'{self.pretraining_params["num_hidden_layers"]}x{self.pretraining_params["layer_size"]}-{self.pretraining_params["embedding_neurons"]}_Drp_{dr}_Lr_{self.pretraining_params["learning_rate"]}_{time.strftime("run_%d-%H_%M_%S")}'
        eps = [str(x) for x in self.pretraining_params["epochs"]]
        btch = [str(x) for x in self.pretraining_params["batch_sizes"]]
        extra = f'_Eps[{",".join(eps)}]_Batch[{",".join(btch)}]'
        self.log_dir = os.path.join('Logs', self.name + '_' + self.formatted )#+ extra + '/')
        self.bins = None


    @abc.abstractmethod
    def do_binning(self) -> [[]]:
        pass

    @abc.abstractmethod
    def load_model(self):
        pass

    @abc.abstractmethod
    def create_autoencoder(self):
        pass

    def loss_fn(self):
        number_of_samples = self.pretraining_params['number_of_samples']
        factor = self.pretraining_params['abd_weight']
        def loss(y_pred, y_true):
            s = number_of_samples
            y_true_tnfs = y_true[:, :-s]
            y_true_abd = y_true[:, -s:]

            y_pred_tnfs = y_pred[:, :-s]
            y_pred_abd = y_pred[:, -s:]

            # TNF error
            # tnf_diff = tf.abs(y_true_tnfs - y_pred_tnfs)

            # tnf_err = tf.reduce_mean(tf.reduce_sum(tnf_diff, axis=1))
            tnf_err = tf.losses.MAE(y_pred_tnfs, y_true_tnfs)

            # ABUNDANCE error
            # abd_diff = tf.abs(y_true_abd - y_pred_abd)
            # abd_err = tf.reduce_mean(tf.reduce_sum(abd_diff, axis=1))
            abd_err = tf.losses.MAE(y_pred_abd, y_true_abd)
            # total_abd_err = tf.reduce_sum(-(tf.math.log(y_pred_abd + 1e-9)) * y_true_abd, axis=1)

            # ratio = np.ceil(num_tnfs / s)
            # loss = tnf_err/ s + abd_err
            # loss = (tnf_err / (s + num_tnfs)) + abd_err
            loss_value = tnf_err/(s*factor) + abd_err
            return loss_value
        return loss

    def sensitivity(self, tnf_feature_size=136, abundance_feature_size=2):

        dataset_variable = tf.Variable(self.x_train)
        with tf.GradientTape() as tape:
            reconstructed = self.autoencoder(dataset_variable, training=True)
            #loss = tf.reduce_mean(tf.keras.losses.MAE(dataset_variable, reconstructed))
            loss = self.loss_fn()(dataset_variable, reconstructed)
        gradients = tape.gradient(loss, dataset_variable)
        gradients = tf.reduce_mean(tf.abs(gradients), 0)
        number_of_features = self.feature_matrix.shape[1]

        tnf_feature = [f'TNF\n{e}' for e in range(1, tnf_feature_size + 1)]
        abundance_feature = [f'ABD\n{e}' for e in range(1, number_of_features - (tnf_feature_size - 1))]
        shape = int(np.ceil(np.sqrt(number_of_features)))
        labels = tnf_feature + abundance_feature
        labels += list(range(0, (shape * shape) - number_of_features))
        mask = np.zeros(shape * shape, dtype=bool)
        for index in range(number_of_features, shape * shape):
            mask[index] = True
        mask = mask.reshape(shape, shape)
        labels = np.asarray(labels).reshape(shape, shape)
        gradient_list = np.asarray(gradients.numpy()).tolist()
        max_gradient = max(gradient_list)
        min_gradient = min(gradient_list)
        for _ in range((shape * shape) - number_of_features):
            gradient_list.append(0.0)
        np_gradients = np.asarray(gradient_list).reshape(shape, shape)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax = sns.heatmap(np_gradients, ax=ax, vmin=min_gradient, vmax=max_gradient, annot=labels, fmt='', mask=mask)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        # plt.show()
        plt.savefig(os.path.join(self.log_dir, 'sensitivity.png'))
        return gradients

    def get_assignments(self, include_outliers=False):

        if include_outliers is False:
            outlier_mask = (self.bins != -1)
        else:
            outlier_mask = np.ones(self.bins.shape[0], dtype=bool)

        try:
            if type(self.contig_ids) is not type(self.bins):
                temp_contigs = np.array(self.contig_ids)
                result = np.vstack([temp_contigs[outlier_mask], self.bins[outlier_mask]])
            else:
                result = np.vstack([self.contig_ids[outlier_mask], self.bins[outlier_mask]])
        except:
            print('Could not combine contig ids with bin assignment')
            print('\nContig IDs:', len(self.contig_ids), '\nBins:', len(self.bins))
            print('contig id: ', str(type(self.contig_ids)), ' bins: ',str(type(self.bins)))
            sys.exit('\nProgram finished without results')
        return result

    def extract_encoder(self):
        input = self.autoencoder.get_layer('input').output
        latent_output = self.autoencoder.get_layer('latent_layer').output
        encoder = keras.Model(input, latent_output, name="encoder")
        return encoder

    def pretraining(self, callbacks=[]):

        file_writer = tf.summary.create_file_writer(self.log_dir) if self.debug else None

        train_loss = []
        x_train = tf.data.Dataset.from_tensor_slices(self.x_train)
        validation_loss = 0
        trained_samples = 0
        current_epoch = 0
        random.seed(2)
        epochs = self.pretraining_params['epochs']
        batch_sizes = self.pretraining_params['batch_sizes']
        print(f'Training for {sum(epochs)} epochs')

        for epochs_to_run, batch_size in zip(epochs, batch_sizes):
            epoch_seed = random.randint(0, 40000)
            training_set = x_train.shuffle(buffer_size=40000, seed=epoch_seed).repeat().batch(batch_size)

            for epoch in range(epochs_to_run):
                for batch in training_set:
                    trained_samples += batch_size
                    train_loss.append(self.autoencoder.train_on_batch(batch, batch))
                    if trained_samples >= self.training_set_size:
                        trained_samples = 0
                        break

                # Epoch over - get metrics

                mean_train_loss = np.mean(train_loss)

                print(f'Epoch\t{current_epoch + 1}\t\tTraining_loss:{mean_train_loss:.6f}')
                if self.debug:
                    with file_writer.as_default():
                        tf.summary.scalar('Training loss', mean_train_loss, step=current_epoch + 1)
                        tf.summary.scalar('Validation loss', validation_loss, step=current_epoch + 1)
                train_loss.clear()
                if (current_epoch + 1) % self.pretraining_params['callback_interval'] == 0:
                    for callback in callbacks:
                        callback.on_epoch_end(epoch=current_epoch + 1)
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

        rerouted_autoencoder.compile(optimizer=opt, loss=[self.clust_params['reconst_loss'], GaussianLoss(bandwidth=self.clust_params['gaussian_bandwidth'])],
                                     loss_weights=self.clust_params['loss_weights'])

        print(rerouted_autoencoder.summary())
        return rerouted_autoencoder


    def save_params(self):
        path = os.path.join(self.log_dir,'training_config.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path,'w') as config_file:
            config = {
                'pretraining_params': self.pretraining_params,
                'clust_params': self.clust_params,
            }
            json.dump(config, config_file)


class Stacked_Binner(Binner):

    def __init__(self, name, contig_ids, feature_matrix, labels=None, x_train=None,
                 x_valid=None, train_labels=None,validation_labels=None, pretraining_params=None, clust_params=None, debug=False):
        super().__init__(name=name, contig_ids=contig_ids, feature_matrix=feature_matrix, labels=labels,
                         x_train=x_train, x_valid=x_valid, train_labels=train_labels, valid_labels=validation_labels,
                         pretraining_params=pretraining_params, clust_params=clust_params, debug=debug)

    def create_autoencoder(self):
        # get params from param dict
        input_shape = self.input_shape
        lr = self.pretraining_params['learning_rate']
        reconst_loss = self.pretraining_params['reconst_loss']
        activation_fn = self.pretraining_params['activation_fn']
        regularizer = self.pretraining_params['regularizer']
        init_fn = self.pretraining_params['initializer']
        no_neurons_hidden = self.pretraining_params['layer_size']
        num_hidden_layers = self.pretraining_params['num_hidden_layers']
        no_neurons_embedding = self.pretraining_params['embedding_neurons']
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
            layer = tf.keras.layers.Dropout(drop_rate)(layer) if drop and layer_index != 0 else layer
            layer = tf.keras.layers.BatchNormalization()(layer) if bn else layer
            layer = tf.keras.layers.Dense(units=no_neurons_hidden, activation=activation_fn, kernel_initializer=init_fn,
                                          kernel_regularizer=regularizer,
                                          name=f'encoding_layer_{layer_index}')(layer)
        # TODO extra large layer before embed
        # layer = tf.keras.layers.Dense(units=400, activation=activation_fn, kernel_initializer=init_fn,
        #                                  kernel_regularizer=regularizer,
        #                                  name=f'encoding_layer_THICC')(layer)

        # create embedding layer
        layer = tf.keras.layers.BatchNormalization()(layer) if bn else layer
        embedding_layer = tf.keras.layers.Dense(units=no_neurons_embedding, kernel_initializer=init_fn,
                                                name='latent_layer')(layer)

        # Create decoding layers
        layer = embedding_layer

        # TODO extra large la  yer before embed
        # layer = tf.keras.layers.Dense(units=400, activation=activation_fn, kernel_initializer=init_fn,
        #                              kernel_regularizer=regularizer,
        #                              name=f'Decoding_layer_THICC')(layer)

        for layer_index in range(num_hidden_layers):
            layer = tf.keras.layers.Dropout(drop_rate)(layer) if drop and layer_index != 0 else layer
            layer = tf.keras.layers.BatchNormalization()(layer) if bn else layer
            layer = tf.keras.layers.Dense(units=no_neurons_hidden, activation=activation_fn, kernel_initializer=init_fn,
                                          kernel_regularizer=regularizer,
                                          name=f'decoding_layer_{layer_index}')(layer)

        layer = tf.keras.layers.Dropout(drop_rate)(layer) if drop else layer
        layer = tf.keras.layers.BatchNormalization()(layer) if bn else layer
        decoder_output = tf.keras.layers.Dense(units=input_shape, kernel_initializer=init_fn,
                                               name='Decoder_output_layer')(layer)
        '''
        tnf = decoder_output[:,:-5]
        abd = decoder_output[:,-5:]

        sm_abd = tf.keras.activations.softmax(abd)

        concat = tf.keras.layers.Concatenate(axis=1)([tnf, sm_abd])
        '''

        stacked_ae = tf.keras.Model(encoder_input, decoder_output, name='autoencoder')

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)



        stacked_ae.compile(optimizer=optimizer, loss=self.loss_fn())
        #stacked_ae.compile(optimizer=optimizer, loss='mae')
        print(stacked_ae.summary())
        self.autoencoder = stacked_ae
        self.encoder = self.extract_encoder()

        return stacked_ae

    def load_model(self, cluster_model=False):
        if cluster_model:
            self.autoencoder = tf.keras.models.load_model('autoencoder', custom_objects={"GaussianLoss": GaussianLoss})
        else:
            self.autoencoder = tf.keras.models.load_model('autoencoder')
        self.encoder = self.extract_encoder()
        print("Model loaded")

    def do_binning(self, load_model=False, load_clustering_AE=True):
        if self.debug:

            self.save_params()
            '''Method is ALMOST identical to Sparse_AE... uses Stacked AE instead'''
            if load_model:
                self.load_model()

            else:
                self.autoencoder = self.create_autoencoder()
                self.encoder = self.extract_encoder()
                print(self.encoder)
                callback_projector = ProjectEmbeddingCallback(binner=self)
                callback_activity = ActivityCallback(binner=self)
                callback_results = WriteBinsCallback(binner=self)

                self.pretraining(callbacks=[callback_projector, callback_activity, callback_results])

            if load_clustering_AE:
                self.load_model(cluster_model=True)
            else:
                #self.autoencoder = self.include_clustering_loss()
                #self.encoder = self.extract_encoder()
                #callback_results = WriteBinsCallback(binner=self, clustering_ae=True)
                #callback_projector.prefix_name = 'DeepClustering_'
                #self.fit_clustering([callback_results])
                print("Completing...")
        else:
            self.autoencoder = self.create_autoencoder()
            self.encoder = self.extract_encoder()
            self.pretraining()
            run_hdbscan(self)
            return self.bins

    def fit_clustering_backup(self, callbacks):
        file_writer = tf.summary.create_file_writer(self.log_dir)
        x = self.x_train
        eps = self.clust_params['eps']
        best_loss = 1e10
        no_improvemnt_epochs = 0
        min_samples = self.clust_params['min_samples']
        epochs = self.clust_params['epochs']

        for epoch in range(epochs):
            print(f'Current epoch: {epoch + 1} of total epochs: {epochs}')

            # 2. encode all data
            encoded_data_full = self.encoder.predict(x)

            # 3. cluster

            current_time = time.strftime('%H:%M:%S')
            print(current_time)

            hdbscan_instance = hdbscan.HDBSCAN(min_cluster_size=self.clust_params['min_cluster_size'], min_samples=self.clust_params['min_samples'], core_dist_n_jobs=36)
            all_assignments = hdbscan_instance.fit_predict(encoded_data_full)

            centroid_dict = {}
            assignment_set = set(all_assignments)
            assignments_unique_sorted = sorted(list(assignment_set))
            for cluster_label in assignments_unique_sorted:
                # if outlier
                if cluster_label == -1:
                    continue

                cluster_mask = (all_assignments == cluster_label)
                encoded_cluster = encoded_data_full[cluster_mask]
                centroid = np.mean(encoded_cluster, axis=0)
                centroid_dict[cluster_label] = centroid

            centroids = np.vstack(list(centroid_dict.values()))
            corrected_assignments = all_assignments.copy()
            centroids_of_assignments = []

            for index, (contig, cluster_label) in enumerate(zip(encoded_data_full, all_assignments)):
                if cluster_label == -1:
                    dists = np.sqrt(np.sum(((centroids - contig) ** 2), axis=1))
                    label_of_shortest_dist = np.argmin(dists)

                    centroids_of_assignments.append(centroid_dict[label_of_shortest_dist])
                    corrected_assignments[index] = label_of_shortest_dist
                else:
                    centroids_of_assignments.append(centroid_dict[cluster_label])

            complete_time = time.strftime('%H:%M:%S')
            print(complete_time)

            print(Counter(all_assignments))
            no_clusters = max(all_assignments)
            print(f'No. clusters: {no_clusters}')

            list_of_losses = self.autoencoder.train_on_batch(x=[x], y=[x, np.array(centroids_of_assignments)])

            combined_loss = list_of_losses[0]
            reconstruction_loss = list_of_losses[1]
            clustering_loss = list_of_losses[2]

            print(
                f'Epoch\t{epoch + 1}\t\tReconstruction_loss:{reconstruction_loss:.4f}\tClustering_loss:{clustering_loss:.4f}\tTotal_loss:{combined_loss:.4f}')
            with file_writer.as_default():
                tf.summary.scalar('DC_Reconstruction loss', reconstruction_loss, step=epoch + 1)
                tf.summary.scalar('DC_Clustering loss', clustering_loss, step=epoch + 1)
                tf.summary.scalar('DC_Total loss', combined_loss, step=epoch + 1)
            if combined_loss < best_loss:
                best_loss = combined_loss
                no_improvemnt_epochs = 0
                if epoch > 100:
                    self.binner.clustering_autoencoder.save(
                        os.path.join(self.binner.log_dir, 'clustering_autoencoder.h5'))
            else:
                no_improvemnt_epochs += 1
            if (epoch + 1) % self.clust_params['callback_interval'] == 0:
                for callback in callbacks:
                    callback.on_epoch_end(epoch + 1,logs=all_assignments)
            if no_improvemnt_epochs >= 20:
                break
        for callback in callbacks:
            callback.on_train_end()

    def fit_clustering(self, callbacks):
        file_writer = tf.summary.create_file_writer(self.log_dir)
        x = self.x_train
        epochs = self.clust_params['epochs']
        eps = self.clust_params['eps']
        best_loss = 1e10
        no_improvemnt_epochs = 0
        #optimizer_weights = getattr(self.autoencoder.optimizer, 'weights')
        min_samples = self.clust_params['min_samples']
        optimizer = self.autoencoder.optimizer
        #optimizer = tf.keras.optimizers.Adam(self.clust_params['learning_rate'])
        #optimizer.set_weights(optimizer_weights)
        for epoch in range(epochs):
            print(f'Current epoch: {epoch + 1} of total epochs: {epochs}')

            # 2. encode all data
            encoded_data_full = self.encoder.predict(x)

            # 3. cluster

            current_time = time.strftime('%H:%M:%S')
            print(current_time)

            hdbscan_instance = hdbscan.HDBSCAN(min_cluster_size=self.clust_params['min_cluster_size'],
                                               min_samples=self.clust_params['min_samples'], core_dist_n_jobs=36)
            all_assignments = hdbscan_instance.fit_predict(encoded_data_full)

            centroid_dict = {}
            variance_dict = {}
            assignment_set = set(all_assignments)
            assignments_unique_sorted = sorted(list(assignment_set))
            for cluster_label in assignments_unique_sorted:
                # if outlier
                if cluster_label == -1:
                    continue

                cluster_mask = (all_assignments == cluster_label)
                encoded_cluster = encoded_data_full[cluster_mask]
                centroid = np.mean(encoded_cluster, axis=0)
                centroid_dict[cluster_label] = centroid
                dists = np.sum(((encoded_cluster - centroid) ** 2), axis=1)
                variance = np.sum(dists)/encoded_cluster.shape[0]
                variance_dict[cluster_label] = variance

            centroids = np.vstack(list(centroid_dict.values()))
            variances = np.hstack(list(variance_dict.values()))
            corrected_assignments = all_assignments.copy()
            centroids_of_assignments = []
            variances_of_assignments = []

            for index, (contig, cluster_label) in enumerate(zip(encoded_data_full, all_assignments)):
                if cluster_label == -1:
                    dists = np.sum(((centroids - contig) ** 2), axis=1)
                    gaussian_dists = np.exp(-dists / (2 * variances))

                    label_of_shortest_dist = np.argmin(gaussian_dists)
                    centroids_of_assignments.append(centroid_dict[label_of_shortest_dist])
                    corrected_assignments[index] = label_of_shortest_dist
                    variances_of_assignments.append(variance_dict[label_of_shortest_dist])
                else:
                    centroids_of_assignments.append(centroid_dict[cluster_label])
                    variances_of_assignments.append(variance_dict[cluster_label])

            variances_of_assignments_tensor = tf.convert_to_tensor(variances_of_assignments, dtype=tf.float32)
            centroids_of_assignments_tensor = tf.convert_to_tensor(centroids_of_assignments, dtype=tf.float32)

            with tf.GradientTape() as tape:

                # Reconstruction loss
                reconstructed = self.autoencoder(x, training=True)
                reconstruction_loss = self.clust_params['loss_weights'][0] * self.loss_fn()(reconstructed, x)

                # Clustering loss
                dists = tf.reduce_sum(((encoded_data_full - centroids_of_assignments_tensor) ** 2), axis=1)

                gaussian_weights = tf.exp(-dists / (2 * variances_of_assignments_tensor))
                clustering_loss = (gaussian_weights * dists) * self.clust_params['loss_weights'][1]

                loss = tf.add_n([reconstruction_loss + clustering_loss] + self.autoencoder.losses)

            gradients = tape.gradient(loss, self.autoencoder.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.autoencoder.trainable_variables))




            complete_time = time.strftime('%H:%M:%S')
            print(complete_time)

            print(Counter(all_assignments))
            no_clusters = max(all_assignments)
            print(f'No. clusters: {no_clusters}')
            weighted_reconstruction_loss = self.clust_params['loss_weights'][0] * np.mean(reconstruction_loss)
            weighted_clustering_loss = self.clust_params['loss_weights'][1] * np.mean(clustering_loss)
            combined_loss = weighted_reconstruction_loss + weighted_clustering_loss


            print(
                f'Epoch\t{epoch + 1}\t\tReconstruction_loss:{weighted_reconstruction_loss:.4f}\tClustering_loss:{weighted_clustering_loss:.4f}\tTotal_loss:{combined_loss:.4f}')
            with file_writer.as_default():
                tf.summary.scalar('DC_Reconstruction loss', weighted_reconstruction_loss, step=epoch + 1)
                tf.summary.scalar('DC_Clustering loss', weighted_clustering_loss, step=epoch + 1)
                tf.summary.scalar('DC_Total loss', combined_loss, step=epoch + 1)
            if combined_loss < best_loss:
                best_loss = combined_loss
                no_improvemnt_epochs = 0
                if epoch > 100:
                    self.binner.clustering_autoencoder.save(
                        os.path.join(self.binner.log_dir, 'clustering_autoencoder'))
            else:
                no_improvemnt_epochs += 1
            if (epoch + 1) % self.clust_params['callback_interval'] == 0:
                for callback in callbacks:
                    callback.on_epoch_end(epoch + 1)
            if no_improvemnt_epochs >= 100:
                break
        for callback in callbacks:
            callback.on_train_end()


class Sparse_Binner(Stacked_Binner):

    def create_autoencoder(self):

        input_shape = self.input_shape
        lr = self.pretraining_params['learning_rate']
        reconst_loss = self.pretraining_params['reconst_loss']
        activation_fn = self.pretraining_params['activation_fn']
        regularizer = self.pretraining_params['regularizer']
        init_fn = self.pretraining_params['initializer']
        no_neurons_hidden = self.pretraining_params['layer_size']
        num_hidden_layers = self.pretraining_params['num_hidden_layers']
        no_neurons_embedding = self.pretraining_params['embedding_neurons']
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



        stacked_ae.compile(optimizer=optimizer, loss=self.loss_fn())

        print(stacked_ae.summary())

        return stacked_ae

    def load_model(self):
        self.autoencoder = tf.keras.models.load_model('autoencoder.h5', custom_objects={
            "KLDivergenceRegularizer": KLDivergenceRegularizer})
        self.encoder = self.extract_encoder()
        print("Model loaded")

    '''def do_binning(self, load_model=False, load_clustering_AE=True):

        if load_model:
            self.load_model()
        else:
            self.autoencoder = self.create_sparse_AE()
            callback_projector = ProjectEmbeddingCallback(binner=self)
            callback_save = SaveBestModelCallback(binner=self)
            callback_activity = ActivityCallback(binner=self)

            self.pretraining(callbacks=[callback_projector, callback_activity, callback_save])
            print("AE saved")

        self.encoder = self.extract_encoder()
        self.encoder.save('encoder.h5')
        print("Encoder saved")

        if load_clustering_AE:
            self.autoencoder = tf.keras.models.load_model("clustering_AE.h5", custom_objects={"KLDivergenceRegularizer": KLDivergenceRegularizer})
            self.encoder = self.extract_encoder()
        else:
            self.autoencoder = self.include_clustering_loss()
            callback_projector.prefix_name = 'DeepCluster_'
            callback_results = WriteBinsCallback(binner=self)

            self.fit_clustering([callback_projector, callback_activity, callback_results])
            print("clustering_AE saved")

        self.bins = self.final_DBSCAN_clustering()
        return self.bins
        '''


class Contractive_Binner(Stacked_Binner):
    def __init__(self, name, contig_ids, feature_matrix, labels=None, x_train=None,
                 x_valid=None, train_labels=None,validation_labels=None, pretraining_params=None, clust_params=None):
        super().__init__(name=name, contig_ids=contig_ids, feature_matrix=feature_matrix, labels=labels,
                         x_train=x_train, x_valid=x_valid, train_labels=train_labels, validation_labels=validation_labels,
                         pretraining_params=pretraining_params, clust_params=clust_params)

    def do_binning(self, load_model=False, load_clustering_AE=True):
        '''Method is ALMOST identical to Sparse_AE... uses Stacked AE instead'''
        if load_model:
            self.autoencoder = tf.keras.models.load_model('autoencoder.h5')
        else:
            self.autoencoder = self.create_autoencoder()
            #callback_projector = ProjectEmbeddingCallback(binner=self)
            callback_results = WriteBinsCallback(binner=self)
            callback_activity = ActivityCallback(binner=self)

            self.pretraining(callbacks=[callback_activity, callback_results])

        self.encoder = self.extract_encoder()

        if load_clustering_AE:
            self.autoencoder = tf.keras.models.load_model("clustering_AE.h5")
            self.encoder = self.extract_encoder()
        else:
            print("Skipping DC")
            #self.autoencoder = self.include_clustering_loss()
            #callback_projector.prefix_name = 'DeepClustering_'
            #callback_save = SaveBestModelCallback(binner=self, clustering=True)
            #self.fit_dbscan([callback_projector, callback_save])
            #self.autoencoder.save('clustering_autoencoder.h5')
            #print("clustering_AE saved")

        return self.bins

    def extract_decoder(self):
        embedding_size = self.autoencoder.get_layer('latent_layer').units

        decoder_input = tf.keras.layers.Input(shape=(embedding_size,), name="input")
        var = decoder_input
        found_middle = False
        for index, layer in enumerate(self.autoencoder.layers):
            if layer.name == 'latent_layer' or found_middle:
                layer_no = index
                if found_middle is False:
                    found_middle = True
                    continue
                var = self.autoencoder.layers[layer_no](var)
        decoder = tf.keras.Model(decoder_input, var, name="decoder")
        return decoder

    def pretraining(self, callbacks):


        def loss_fn(y_pred, y_true):
            s = self.pretraining_params['number_of_samples']
            y_true_tnfs = y_true[:, :-s]
            y_true_abd = y_true[:, -s:]

            y_pred_tnfs = y_pred[:, :-s]
            y_pred_abd = y_pred[:, -s:]

            # TNF error
            #tnf_diff = tf.abs(y_true_tnfs - y_pred_tnfs)

            #tnf_err = tf.reduce_mean(tf.reduce_sum(tnf_diff, axis=1))
            tnf_err = tf.losses.MAE(y_pred_tnfs, y_true_tnfs)

            # ABUNDANCE error
            #abd_diff = tf.abs(y_true_abd - y_pred_abd)
            #abd_err = tf.reduce_mean(tf.reduce_sum(abd_diff, axis=1))
            abd_err = tf.losses.MAE(y_pred_abd, y_true_abd)
            # total_abd_err = tf.reduce_sum(-(tf.math.log(y_pred_abd + 1e-9)) * y_true_abd, axis=1)

            # ratio = np.ceil(num_tnfs / s)
            # loss = tnf_err/ s + abd_err
            # loss = (tnf_err / (s + num_tnfs)) + abd_err
            loss_value = tnf_err/(s*self.pretraining_params['abd_weight']) + abd_err
            return loss_value

        file_writer = tf.summary.create_file_writer(self.log_dir)
        jacobian_losses = []
        reconstruction_losses = []
        combined_losses = []
        random.seed(2)
        x_train = tf.data.Dataset.from_tensor_slices(self.x_train)
        loss_function = tf.keras.losses.get(self.autoencoder.loss)
        optimizer = tf.optimizers.Adam(self.pretraining_params['learning_rate'])
        trained_samples = 0
        current_epoch = 0
        encoder = self.extract_encoder()
        decoder = self.extract_decoder()
        weight = 1e-4

        for epochs_to_run, batch_size in zip(self.pretraining_params['epochs'],
                                             self.pretraining_params['batch_sizes']):
            epoch_seed = random.randint(0, 40000)
            for epoch in range(epochs_to_run):
                training_set = x_train.shuffle(buffer_size=40000, seed=epoch_seed).repeat().batch(batch_size)
                for batch in training_set:
                    trained_samples += batch_size

                    with tf.GradientTape() as bp_tape:

                        with tf.GradientTape() as jacobian_tape:
                            jacobian_tape.watch(batch)
                            embedding = encoder(batch, training=True)

                        jacobian_batch = jacobian_tape.batch_jacobian(embedding, batch)
                        mean_grads = tf.reduce_mean(jacobian_batch, axis=0)
                        jacobian_loss = tf.reduce_sum(mean_grads ** 2)

                        reconstruction = decoder(embedding, training=True)
                        reconstruction_loss = loss_fn(batch, reconstruction)
                        loss = tf.add_n([reconstruction_loss + weight * jacobian_loss] + self.autoencoder.losses)

                    reconstruction_losses.append(reconstruction_loss)
                    jacobian_losses.append(jacobian_loss * weight)
                    combined_losses.append(reconstruction_loss + (jacobian_loss*weight))

                    gradients = bp_tape.gradient(loss, self.autoencoder.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, self.autoencoder.trainable_variables))

                    if trained_samples >= self.training_set_size:
                        trained_samples = 0
                        break

                # Epoch over - get metrics


                mean_jacobian = np.mean(jacobian_losses)
                mean_reconstruction = np.mean(reconstruction_losses)
                mean_combined = np.mean(combined_losses)
                print(
                    f'Epoch\t{current_epoch + 1}\tReconstruction_loss:{mean_reconstruction:.4f}\tJacobian_loss:{mean_jacobian:.4f}')
                with file_writer.as_default():
                    tf.summary.scalar('Training_jacobian_loss', mean_jacobian, step=current_epoch + 1)
                    tf.summary.scalar('Training_reconstruction_loss', mean_reconstruction, step=current_epoch + 1)
                    tf.summary.scalar('Training_total_loss', mean_combined, step=current_epoch + 1)

                jacobian_losses.clear(), reconstruction_losses.clear(), combined_losses.clear()
                if (current_epoch + 1) % self.pretraining_params['callback_interval'] == 0:
                    for callback in callbacks:
                        callback.on_epoch_end(current_epoch + 1)
                current_epoch += 1
        for callback in callbacks:
            callback.on_train_end()

    def fit_dbscan(self, callbacks):
        x = self.x_train
        y = self.labels
        batch_size = self.clust_params['batch_size']
        epochs = self.clust_params['epochs']
        eps = self.clust_params['eps']
        min_samples = self.clust_params['min_samples']
        encoder = self.extract_encoder()
        decoder = self.extract_decoder()
        weight_jacobian = 1e-4
        weight_clustering = 0.05
        jacobian_losses = []
        reconstruction_losses = []
        clustering_losses = []
        combined_losses = []
        loss_function = tf.keras.losses.get(self.clust_params['clust_loss'])
        optimizer = self.pretraining_params['optimizer'](self.pretraining_params['learning_rate'])

        file_writer = tf.summary.create_file_writer(self.log_dir)

        reconstruction_loss = []
        clustering_loss = []
        combined_loss = []

        tensorboard = keras.callbacks.TensorBoard(
            log_dir=self.log_dir,
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

                hdbscan_instance = hdbscan.HDBSCAN(min_cluster_size=4, core_dist_n_jobs=36)
                all_assignments = hdbscan_instance.fit_predict(encoded_data_full)

                # Todo this should be np array
                #all_assignments = np.array(dbscan_instance.labels_)
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


                print(Counter(all_assignments))
                no_clusters = max(all_assignments)
                print(f'No. clusters: {no_clusters}')
                batch = tf.constant(batch)
                with tf.GradientTape() as bp_tape:

                    with tf.GradientTape() as jacobian_tape:
                        jacobian_tape.watch(batch)
                        embedding = encoder(batch, training=True)

                    jacobian_batch = jacobian_tape.batch_jacobian(embedding, batch)
                    mean_grads = tf.reduce_mean(jacobian_batch, axis=0)
                    jacobian_loss = tf.reduce_sum(mean_grads ** 2)

                    reconstruction = decoder(embedding, training=True)
                    clustering_loss = tf.reduce_mean(loss_function(embedding, batch_centroids))
                    reconstruction_loss = tf.reduce_mean(loss_function(batch, reconstruction))
                    loss = tf.add_n([(reconstruction_loss + weight_jacobian * jacobian_loss) + weight_clustering * clustering_loss] + self.autoencoder.losses)

                reconstruction_losses.append(reconstruction_loss)
                jacobian_losses.append(jacobian_loss * weight_jacobian)
                combined_losses.append((reconstruction_loss + jacobian_loss * weight_jacobian) + (clustering_loss*weight_clustering))
                clustering_losses.append(clustering_loss*weight_clustering)

                gradients = bp_tape.gradient(loss, self.autoencoder.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.autoencoder.trainable_variables))


            print(
                f'Epoch\t{i + 1}\t\tReconstruction_loss:{np.mean(reconstruction_losses):.4f}\tClustering_loss:{np.mean(clustering_losses):.4f}\tClustering_loss:{np.mean(jacobian_losses):.4f}\tTotal_loss:{np.mean(combined_losses):.4f}')
            with file_writer.as_default():
                tf.summary.scalar('DC_Reconstruction loss', np.mean(reconstruction_losses), step=i + 1)
                tf.summary.scalar('DC_Clustering loss', np.mean(clustering_losses), step=i + 1)
                tf.summary.scalar('DC_Jacobian loss', np.mean(jacobian_losses), step=i + 1)
                tf.summary.scalar('DC_Total loss', np.mean(combined_losses), step=i + 1)
            reconstruction_losses.clear(), clustering_losses.clear(), combined_losses.clear(), jacobian_losses.clear()

            for callback in callbacks:
                callback.on_epoch_end(i + 1)
        for callback in callbacks:
            callback.on_train_end()
        tensorboard.on_train_end(None)


class GaussianLoss(tf.keras.losses.Loss):
    def __init__(self, bandwidth=1, **kwargs):
        self.bandwidth = bandwidth
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        abs_err = tf.abs(error)
        squared_err = tf.square(abs_err)
        neg_squared_err = -squared_err

        gaussian_weight = tf.exp(neg_squared_err/(2 * self.bandwidth))

        return gaussian_weight * (0.5 * squared_err)

    def get_config(self):
        return {"bandwidth": self.bandwidth}


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


def run_hdbscan(binner):
    embedding = binner.encoder.predict(binner.x_train)
    hdbscan_instance = hdbscan.HDBSCAN(min_cluster_size=binner.clust_params['min_cluster_size'],
                                       min_samples=binner.clust_params['min_samples'], core_dist_n_jobs=-1,
                                       metric='euclidean')
    all_assignments = hdbscan_instance.fit_predict(embedding)
    binner.bins = all_assignments

def create_binner(binner_type, contig_ids, feature_matrix, labels=None, x_train=None, x_valid=None, train_labels=None,
                  validation_labels=None, pretraining_params=None, clust_params=None, debug=False):
    binner = binner_dict[binner_type]

    binner_instance = binner(name=binner_type, contig_ids=contig_ids, feature_matrix=feature_matrix, labels=labels,
                                  x_train=x_train, x_valid=x_valid,train_labels=train_labels, validation_labels=validation_labels, pretraining_params=pretraining_params, clust_params=clust_params, debug=False)
    return binner_instance


class ProjectEmbeddingCallback(tf.keras.callbacks.Callback):

    def __init__(self, binner, log_dir=None, prefix_name=False):
        super().__init__()
        self.binner = binner
        self.prefix_name = '' if prefix_name == False else prefix_name
        if log_dir == None:
            self.log_dir = self.binner.log_dir
        else:
            self.log_dir = log_dir
        self.embeddings = []

    def on_epoch_end(self, epoch, logs=None):

        embedding_mean_substracted_unit_norm = self.binner.encoder.predict(self.binner.x_train)
        embedding_mean = np.mean(embedding_mean_substracted_unit_norm, axis=0)
        embedding_mean_substracted_unit_norm -= embedding_mean
        embedding_mean_substracted_unit_norm = normalize(embedding_mean_substracted_unit_norm, axis=1, norm='l2')

        data_to_project = self.project_data(embedding=embedding_mean_substracted_unit_norm, epoch=epoch)
        data_variable = tf.Variable(data_to_project, name=f'{self.prefix_name}epoch{epoch}')
        self.embeddings.append(data_variable)

    def on_train_end(self, logs=None):
        config = projector.ProjectorConfig()
        if len(self.embeddings) != 0:
            saver = tf.compat.v1.train.Saver(var_list=self.embeddings, save_relative_paths=True)
            saver.save(None, os.path.join(self.log_dir, f'embeddings.ckpt'))
            for e in self.embeddings:
                embedding = config.embeddings.add()
                embedding.tensor_name = e.name
                embedding.metadata_path = "metadata.tsv"
            projector.visualize_embeddings(self.log_dir, config)


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

        return data_to_project


class ActivityCallback(tf.keras.callbacks.Callback):

    def __init__(self, binner):
        self.binner = binner
        self.file_writer = tf.summary.create_file_writer(binner.log_dir)
        super().__init__()

    def on_train_end(self):
        pass

    def on_epoch_end(self, epoch, logs=None):
        print("Making activation figures")
        encoder = self.binner.extract_encoder()
        activations = encoder.predict(self.binner.x_train)
        activations_flatten = activations.flatten()
        #activations_percentages = activations.flatten()/activations.size
        number_of_activations = activations.size

        #Creating figure of activities
        figure, ax = plt.subplots()
        y_vals, x_vals, e_ = ax.hist(activations_flatten, edgecolor='black')
        ax.set_title("Embedding activations")
        ax.set_xlabel("Activations")
        ax.set_ylabel("% Activation")

        y_max = round((max(y_vals) / number_of_activations) + 0.01, 2)
        y_max = 1
        ax.set_yticks(ticks=np.arange(0.0, y_max * number_of_activations, 0.1 * number_of_activations,))
        ax.set_ylim(ax.get_yticks()[0], ax.get_yticks()[-1])
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=number_of_activations))
        with self.file_writer.as_default():
            tf.summary.image("Activation histogram", plot_to_image(figure), step=epoch)


        #Creating figure of neurons
        mean_activities = tf.reduce_mean(activations, axis=0).numpy()
        number_neurons = mean_activities.size
        figure, ax = plt.subplots()
        y_vals, x_vals, e_ = ax.hist(mean_activities, edgecolor='black')
        ax.set_title("Activity neuron")
        ax.set_xlabel("Neuron Mean Activation")
        ax.set_ylabel("% Neurons")
        y_max = round((max(y_vals) / number_neurons) + 0.01, 2)
        ax.set_yticks(ticks=np.arange(0.0, y_max * number_neurons, 0.05 * number_neurons, ))
        ax.set_ylim(ax.get_yticks()[0], ax.get_yticks()[-1])
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=number_neurons))
        with self.file_writer.as_default():
            tf.summary.image("Neuron histogram", plot_to_image(figure), step=epoch)
        print("wrote images")


class WriteBinsCallback(tf.keras.callbacks.Callback):
    def __init__(self, binner, clustering_ae=False):
        super().__init__()
        self.binner = binner
        if clustering_ae:
            self.prefix = f'DC_{binner.name}'
        else:
            self.prefix =f'{binner.name}'
        self.clustering_ae = clustering_ae

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            all_assignments = logs
        else:
            # Test
            embedding = self.binner.encoder.predict(self.binner.x_train)
            embedding_mean = np.mean(embedding, axis=0)
            embedding -= embedding_mean
            normalized_embedding = normalize(embedding, axis=1, norm='l2')

            hdbscan_instance = hdbscan.HDBSCAN(min_cluster_size=self.binner.clust_params['min_cluster_size'], min_samples=self.binner.clust_params['min_samples'], core_dist_n_jobs=-1, metric='euclidean')
            all_assignments = hdbscan_instance.fit_predict(normalized_embedding)
            self.binner.bins = all_assignments
            bins_without_outliers = self.binner.get_assignments(include_outliers=False)
            data_processor.write_bins_to_file(bins_without_outliers, output_dir=os.path.join(self.binner.log_dir, f'{self.prefix}_Ep:{epoch}_{self.binner.formatted}_'))

            # Original
            #data = self.binner.encoder.predict(self.binner.x_train)
            #hdbscan_instance = hdbscan.HDBSCAN(min_cluster_size=self.binner.clust_params['min_cluster_size'], min_samples=self.binner.clust_params['min_samples'], core_dist_n_jobs=-1)
            #all_assignments = hdbscan_instance.fit_predict(data)
            #self.binner.bins = all_assignments
            #bins_without_outliers = self.binner.get_assignments(include_outliers=False)

            #data_processor.write_bins_to_file(bins_without_outliers, output_dir=os.path.join(self.binner.log_dir, f'{self.prefix}_Ep:{epoch}_{self.binner.formatted}_'))

            #self.binner.autoencoder.save(os.path.join(self.binner.log_dir, f'{self.prefix}Epoch_{epoch}'))


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
    'SPARSE': Sparse_Binner,
    'CONTRACTIVE': Contractive_Binner
}