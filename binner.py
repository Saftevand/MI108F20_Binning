import clustering_methods
import abc
import data_processor
import tensorflow as tf
from tensorflow import keras
import numpy as np
from clustering_layer_xifeng import ClusteringLayer
from sklearn.cluster import KMeans
from time import time
import sys



class Binner(abc.ABC):
    def __init__(self, contig_ids, clustering_method, split_value, log_dir, feature_matrix=None):
        self.feature_matrix = feature_matrix
        self.bins = None
        self.contig_ids = contig_ids
        self.clustering_method = clustering_method
        self.x_train, self.x_valid = data_processor.get_train_and_validation_data(feature_matrix=self.feature_matrix, split_value=split_value)
        self.encoder = None
        self.full_AE_train_history = None
        self.log_dir = f'{log_dir}/logs'
        self.train_start = None

    @abc.abstractmethod
    def do_binning(self) -> [[]]:
        pass

    def get_assignments(self):
        try:
            result = np.vstack([self.contig_ids, self.bins])
        except:
            try:
                print('Error: \nvstack failed! \nTrying hstack')
                result = np.hstack([self.contig_ids, self.bins])
                print(result[0])
                print('Success')
            except:
                print('Error: \nhstack failed!')
                print('Could not combine contig ids with bin assignment')
                sys.exit('Program finished without results')
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
        self.log_dir = f'{self.log_dir}/SAE'

    def do_binning(self):
        self.set_train_timestamp()
        self.full_AE_train_history, self.full_autoencoder = self.train()
        return self.clustering_method.do_clustering(dataset=self.encoder.predict(self.feature_matrix), contig_names=self.contig_ids)

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

    def do_binning(self, init='glorot_uniform', pretrain_optimizer='adam', n_clusters=10, update_interval=140,
                   pretrain_epochs=20, batch_size=128, tolerance_threshold=1e-3, max_iterations=100, true_bins=None):

        self.set_train_timestamp()
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


def create_binner(split_value, binner_type, clustering_method, contig_ids, feature_matrix, log_dir):
    clustering_method = get_clustering(clustering_method)

    binner_type = get_binner(binner_type)

    binner_instance = binner_type(split_value=split_value, contig_ids=contig_ids, clustering_method=clustering_method(),
                                  feature_matrix=feature_matrix, log_dir=log_dir)
    return binner_instance


def get_binner(binner):
    return binner_dict[binner]


def get_clustering(cluster):
    return clustering_algorithms_dict[cluster]


binner_dict = {
    'DEC': Greedy_pretraining_DEC,
    'SAE': Sequential_Binner,
    'DEC_XIFENG': DEC_Binner_Xifeng
}

clustering_algorithms_dict = {
    'KMeans': clustering_methods.clustering_k_means,
    'Random': clustering_methods.random_cluster,
    'KMeans_gpu': clustering_methods.KMEANS_GPU,
    'DBSCAN_gpu': clustering_methods.DBSCAN_GPU
}
