import clustering_methods
import abc
import data_processor
from tensorflow import keras
import numpy as np
from clustering_layer_xifeng import ClusteringLayer
from sklearn.cluster import KMeans
from time import time

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

nmi = normalized_mutual_info_score
ari = adjusted_rand_score

def acc(y_true, y_pred):
    """
    TODO maybe not needed when up and running -Simon
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


class Binner(abc.ABC):
    def __init__(self, contig_ids, clustering_method, split_value, feature_matrix = None):
        self.feature_matrix = feature_matrix
        self.bins = None #TODO
        self.contig_ids = contig_ids
        self.clustering_method = clustering_method
        self.x_train, x_valid = data_processor.get_train_and_validation_data(feature_matrix=self.feature_matrix, split_value=split_value)
        self.encoder = None

    @abc.abstractmethod
    def do_binning(self) -> [[]]:
        pass

    def get_assignments(self):
        return np.vstack([self.contig_ids, self.bins])


class Sequential_Binner(Binner):
    def __init__(self):
        super().__init__()
        self._input_layer_size = None
        self.decoder = None

    def _encoder(self):
        if (self.encoder is not None):
            return self.encoder
        self._input_layer_size = self.x_train.sample(1).size

        stacked_encoder = keras.models.Sequential([
            keras.layers.Dense(500, activation="selu", input_shape=[self._input_layer_size, ],
                               kernel_initializer=keras.initializers.lecun_normal()),
            keras.layers.Dense(500, activation="selu", kernel_initializer=keras.initializers.lecun_normal()),
            keras.layers.Dense(2000, activation="selu", kernel_initializer=keras.initializers.lecun_normal()),
            keras.layers.Dense(10, kernel_initializer=keras.initializers.lecun_normal()), ])
        self.encoder = stacked_encoder
        return stacked_encoder

    def _decoder(self):
        if (self.decoder is not None):
            return self.decoder

        stacked_decoder = keras.models.Sequential([
            keras.layers.Dense(2000, activation="selu", input_shape=[10],
                               kernel_initializer=keras.initializers.lecun_normal()),
            keras.layers.Dense(500, activation="selu", kernel_initializer=keras.initializers.lecun_normal()),
            keras.layers.Dense(500, activation="selu", kernel_initializer=keras.initializers.lecun_normal()),
            keras.layers.Dense(self._input_layer_size, kernel_initializer=keras.initializers.lecun_normal()), ])
        self.decoder = stacked_decoder
        return stacked_decoder

    def train(self, loss_funciton=keras.losses.mse, optimizer_=keras.optimizers.Adam(lr=0.03), number_of_epoch=50000):
        stacked_ae = keras.models.Sequential([self._encoder(), self._decoder()])
        stacked_ae.compile(loss=loss_funciton,
                           optimizer=optimizer_,
                           metrics=['accuracy'])

        print('Training start')
        history = stacked_ae.fit(x=self.x_train, y=self.x_train, epochs=number_of_epoch,
                                 validation_data=[self.x_valid, self.x_valid])
        print('Training ended')
        return history, stacked_ae

    def extract_features(self, feature_matrix):
        return self.encoder.predict(feature_matrix)


class DEC(Binner):
    def __init__(self, split_value, contig_ids, feature_matrix, clustering_method):
        super().__init__(split_value=split_value, contig_ids=contig_ids, feature_matrix=feature_matrix, clustering_method=clustering_method)
        self.model = None
        self.autoencoder = None
        self.n_clusters = None

    def do_binning(self) -> [[]]:
        pass

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def fit(self, x, y=None, maxiter=2e4, batch_size=258, tolerance_threshold=1e-3,
            update_interval=150, save_dir='./results/temp'):

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

        # Step 2: deep clustering
        # logging file
        import csv
        logfile = open(save_dir + '/dec_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'loss'])
        logwriter.writeheader()

        loss = 0
        index = 0
        index_array = np.arange(x.shape[0])
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                if y is not None:
                    acc = np.round(acc(y, y_pred), 5)
                    nmi = np.round(nmi(y, y_pred), 5)
                    ari = np.round(ari(y, y_pred), 5)
                    loss = np.round(loss, 5)
                    logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, loss=loss)
                    logwriter.writerow(logdict)
                    print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)

                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < tolerance_threshold:
                    print('delta_label ', delta_label, '< tol ', tolerance_threshold)
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break

            # train on batch
            # if index == 0:
            #     np.random.shuffle(index_array)
            idx = index_array[index * batch_size: min((index + 1) * batch_size, x.shape[0])]
            xidx = x[idx]
            pidx = p[idx]
            loss = self.model.train_on_batch(x=xidx, y=pidx)
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

            '''
            # save intermediate model
            if ite % save_interval == 0:
                print('saving model to:', save_dir + '/DEC_model_' + str(ite) + '.h5')
                self.model.save_weights(save_dir + '/DEC_model_' + str(ite) + '.h5')
            '''
            ite += 1

        # save the trained model
        logfile.close()
        print('saving model to:', save_dir + '/DEC_model_final.h5')
        self.model.save_weights(save_dir + '/DEC_model_final.h5')

        return y_pred


class DEC_Binner_Xifeng(DEC):
    def __init__(self, split_value, contig_ids, clustering_method, feature_matrix):
        super().__init__(split_value=split_value, contig_ids=contig_ids, feature_matrix=feature_matrix, clustering_method=clustering_method)
        self._input_layer_size = None

    def do_binning(self, init='glorot_uniform', pretrain_optimizer='adam', n_clusters=10, update_interval=140,
              pretrain_epochs=200, batch_size=128, save_dir='results', tolerance_threshold=1e-3,
              max_iterations=100):

        self.n_clusters = n_clusters
        self.autoencoder, self.encoder = self.define_model(dims=[self.x_train.shape[-1], 500, 500, 2000, 10],
                                                           act='relu', init=init)

        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output)
        self.model = keras.models.Model(inputs=self.encoder.input, outputs=clustering_layer)

        # TODO y er labels... det har vi ikke
        y = None

        self.pretrain(x=self.x_train, y=y, optimizer=pretrain_optimizer,
                      epochs=pretrain_epochs, batch_size=batch_size,
                      save_dir=save_dir)

        self.model.summary()
        t0 = time()
        self.compile(optimizer=keras.optimizers.SGD(0.01, 0.9), loss='kld')
        y_pred = self.fit(self.x_train, y=y, tolerance_threshold=tolerance_threshold, maxiter=max_iterations,
                          batch_size=batch_size,
                          update_interval=update_interval, save_dir=save_dir)
        # TODO hvis vi skal udregne acc til ground truth
        # print('acc:', metrics.acc(y, y_pred))
        print('clustering time: ', (time() - t0))
        self.bins = y_pred

    def pretrain(self, x, y=None, optimizer='adam', epochs=200, batch_size=256, save_dir='results/temp'):
        print('...Pretraining...')
        self.autoencoder.compile(optimizer=optimizer, loss='mse')

        csv_logger = keras.callbacks.CSVLogger(save_dir + '/pretrain_log.csv')
        cb = [csv_logger]
        if y is not None:
            class PrintACC(keras.callbacks.Callback):
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
                    super(PrintACC, self).__init__()

                def on_epoch_end(self, epoch, logs=None):
                    if int(epochs / 10) != 0 and epoch % int(epochs / 10) != 0:
                        return
                    feature_model = keras.models.Model(self.model.input,
                                                       self.model.get_layer(
                                                           'encoder_%d' % (int(len(self.model.layers) / 2) - 1)).output)
                    features = feature_model.predict(self.x)
                    km = KMeans(n_clusters=len(np.unique(self.y)), n_init=20, n_jobs=4)
                    y_pred = km.fit_predict(features)
                    # print()
                    print(' ' * 8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
                          % (acc(self.y, y_pred), nmi(self.y, y_pred)))

            cb.append(PrintACC(x, y))

        # begin pretraining
        t0 = time()
        self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=cb)
        print('Pretraining time: %ds' % round(time() - t0))
        self.autoencoder.save_weights(save_dir + '/ae_weights.h5')
        print('Pretrained weights are saved to %s/ae_weights.h5' % save_dir)
        self.pretrained = True

    def extract_features(self, x):
        return self.encoder.predict(x)

    def predict(self, x):  # predict cluster labels using the output of clustering layer
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)

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


class Greedy_pretraining_DEC(DEC):
    def __init__(self, split_value, clustering_method, feature_matrix, contig_ids):
        super().__init__(split_value=split_value, clustering_method=clustering_method, feature_matrix=feature_matrix, contig_ids=contig_ids)


    def do_binning(self, init='glorot_uniform', pretrain_optimizer='adam', n_clusters=10, update_interval=140,
              pretrain_epochs=200, batch_size=128, save_dir='results', tolerance_threshold=1e-3,
              max_iterations=100):

        self.n_clusters = n_clusters

        # TODO we have no labels
        y = None

        # layerwise and finetuned encoder
        self.greedy_pretraining()

        # Insert clustering layer using KLD error
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output)
        self.model = keras.models.Model(inputs=self.encoder.input, outputs=clustering_layer)

        self.model.compile(optimizer=keras.optimizers.SGD(0.01, 0.9), loss='kld')

        y_pred = self.fit(self.x_train, y=y, tolerance_threshold=tolerance_threshold, maxiter=max_iterations,
                          batch_size=batch_size,
                          update_interval=update_interval, save_dir=save_dir)
        # TODO hvis vi skal udregne acc til ground truth
        # print('acc:', metrics.acc(y, y_pred))
        self.bins = y_pred

    def predict(self, data):
        return self.model.predict(data)

    def greedy_pretraining(self, loss_function=keras.losses.binary_crossentropy, lr=0.01, finetune_epochs=1,
                           pretrain_epochs=1, neuron_list=[500, 10], input_shape=None, dropout_rate=0.2, verbose=1):
        if input_shape is None:
            # feature vector size
            input_shape = self.x_train.shape[1]

        print(f'Adding and training first layer for {pretrain_epochs} epochs')

        neurons_first_layer = neuron_list.pop(0)

        # Create first Encoder decoder pair
        input = keras.layers.Input(shape=(input_shape,))
        dropout_out = keras.layers.Dropout(dropout_rate)(input)
        enc_layer = keras.layers.Dense(neurons_first_layer, activation='selu', input_shape=[input_shape],
                                       kernel_initializer=keras.initializers.lecun_normal())
        enc_out = enc_layer(dropout_out)
        dec_layer = keras.layers.Dense(input_shape, activation='selu', input_shape=[neurons_first_layer],
                                       kernel_initializer=keras.initializers.lecun_normal())
        dec_out = dec_layer(enc_out)

        model = keras.models.Model(inputs=input, outputs=dec_out)

        model.compile(loss=loss_function, optimizer=keras.optimizers.SGD(lr), metrics=['accuracy'])

        #model.fit(x=self.x_train, y=self.x_train, epochs=pretrain_epochs, validation_data=[self.x_valid, self.x_valid], verbose=verbose)

        model.fit(x=self.x_train, y=self.x_train, epochs=pretrain_epochs, verbose = verbose)

        decoder_layer_list = [dec_layer]

        enc_out = enc_layer(input)
        trained_encoder = keras.models.Model(inputs=input, outputs=enc_out)

        # add and train more layers
        full_model = self.add_and_fit_layers(loss_function, lr, pretrain_epochs, decoder_layer_list, neuron_list,
                                             dropout_rate, trained_encoder, input, verbose)

        # finetune_model and return history

        print(f'Finetuning final model for {finetune_epochs} epochs')

        history = full_model.fit(x=self.x_train, y=self.x_train, epochs=finetune_epochs, batch_size=256, verbose=verbose)

        # Saving full autoencoder because why not?
        self.autoencoder = keras.models.clone_model(full_model)
        self.autoencoder.set_weights(full_model.get_weights())
        self.autoencoder.compile(loss=loss_function, optimizer=keras.optimizers.SGD(lr), metrics=['accuracy'])

        # extract encoder from autoencoder
        out = full_model.layers[1](input)
        full_encoder = keras.models.Model(inputs=input, outputs=out)
        full_encoder.compile(loss=loss_function, optimizer=keras.optimizers.SGD(lr), metrics=['accuracy'])

        self.encoder = full_encoder

        return history

    def add_and_fit_layers(self, loss_function, lr, pretrain_epochs, decoder_layer_list, neuron_list, dropout_rate,
                           current_encoder_stack, input, verbose):

        which_layer = len(neuron_list)

        if which_layer == 0:
            return self.combine_encoder_decoder(current_encoder_stack, decoder_layer_list, input, loss_function, lr)

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
            new_enc_layer = keras.layers.Dense(n_neurons, input_shape=[encoded_data_length],
                                               kernel_initializer=keras.initializers.lecun_normal())
            new_enc_out = new_enc_layer(dropout_out)
        else:
            new_enc_layer = keras.layers.Dense(n_neurons, activation='selu', input_shape=[encoded_data_length],
                                               kernel_initializer=keras.initializers.lecun_normal())
            new_enc_out = new_enc_layer(dropout_out)

        # new_enc_layer = keras.layers.Dense(n_neurons, activation='selu', input_shape=[encoded_data_length], kernel_initializer=keras.initializers.lecun_normal())
        # new_enc_out = new_enc_layer(dropout_out)

        new_dec_layer = keras.layers.Dense(encoded_data_length, activation='selu', input_shape=[n_neurons],
                                           kernel_initializer=keras.initializers.lecun_normal())
        new_dec_out = new_dec_layer(new_enc_out)

        model = keras.models.Model(inputs=input_new_layer, outputs=new_dec_out)

        model.compile(loss=loss_function, optimizer=keras.optimizers.SGD(lr), metrics=['accuracy'])

        model.fit(x=encoded_data_train, y=encoded_data_train, epochs=pretrain_epochs, verbose=verbose)

        # puts in the opposite end of append()
        decoder_layer_list.insert(0, new_dec_layer)

        # add new encoder layer to previous encoder layers
        old_out = current_encoder_stack(inputs=input)
        out = new_enc_layer(old_out)

        model = keras.models.Model(inputs=input, outputs=out)
        # skal man compile her?
        model.compile(loss=loss_function, optimizer=keras.optimizers.SGD(lr), metrics=['accuracy'])

        return self.add_and_fit_layers(loss_function, lr, pretrain_epochs, decoder_layer_list, neuron_list,
                                       dropout_rate, model, input, verbose)

    def combine_encoder_decoder(self, encoder, decoder_layers, input, loss_function, lr):

        out = encoder(input)
        for layer in decoder_layers:
            out = layer(out)

        model = keras.models.Model(inputs=input, outputs=out)
        # skal man compile her?
        model.compile(loss=loss_function, optimizer=keras.optimizers.SGD(lr), metrics=['accuracy'])

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


def create_binner(split_value, binner_type, clustering_method, contig_ids, feature_matrix):
    clustering_method = get_clustering(clustering_method)

    binner_type = get_binner(binner_type)

    binner_instance = binner_type(split_value=split_value, contig_ids=contig_ids, clustering_method=clustering_method(), feature_matrix=feature_matrix)
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
    'Random': clustering_methods.random_cluster
}