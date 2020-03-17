import tensorflow as tf
from tensorflow import keras
import numpy as np


class stacked_autoencoder():

    def __init__(self, train, valid, input_layer_size):
        self.x_train = train
        self.x_valid = valid
        self._input_layer_size = input_layer_size
        self.encoder = None
        self.decoder = None

    def _encoder(self):
        if (self.encoder != None):
            return self.encoder

        stacked_encoder = keras.models.Sequential([
            keras.layers.Dense(1000, activation="selu", input_shape=[self._input_layer_size,], kernel_initializer=keras.initializers.lecun_normal()),
            keras.layers.Dense(100, activation="selu", kernel_initializer=keras.initializers.lecun_normal()),
            keras.layers.Dense(2, activation="selu", kernel_initializer=keras.initializers.lecun_normal()), ])
        self.encoder = stacked_encoder
        return stacked_encoder

    def _decoder(self):
        if (self.decoder != None):
            return self.decoder

        stacked_decoder = keras.models.Sequential([
            keras.layers.Dense(100, activation="selu", input_shape=[2], kernel_initializer=keras.initializers.lecun_normal()),
            keras.layers.Dense(1000, activation="selu", kernel_initializer=keras.initializers.lecun_normal()),
            keras.layers.Dense(self._input_layer_size, activation="sigmoid", kernel_initializer=keras.initializers.RandomNormal), ])
        self.decoder = stacked_decoder
        return stacked_decoder

    def train(self, loss_funciton=keras.losses.binary_crossentropy, optimizer_=keras.optimizers.SGD(lr=0.75), number_of_epoch=10):
        stacked_ae = keras.models.Sequential([self._encoder(), self._decoder()])
        stacked_ae.compile(loss=loss_funciton,
                           optimizer=optimizer_,
                           metrics=['accuracy'])

        print('Training start')
        history = stacked_ae.fit(x=self.x_train, y=self.x_train, epochs=number_of_epoch,
                                 validation_data=[self.x_valid, self.x_valid])
        print('Training ended')
        return history, stacked_ae


class DEC_autoencoder():

    def __init__(self, train, valid):
        self.x_train = train
        self.x_valid = valid

    def _encoder(self):
        stacked_encoder = keras.models.Sequential([
            keras.layers.Dense(500, activation="selu", input_shape=[136,], kernel_initializer=keras.initializers.lecun_normal()),
            keras.layers.Dense(500, activation="selu", kernel_initializer=keras.initializers.lecun_normal()),
            keras.layers.Dense(2000, activation="selu", kernel_initializer=keras.initializers.lecun_normal()),
            keras.layers.Dense(10, kernel_initializer=keras.initializers.lecun_normal()), ])
        return stacked_encoder

    def _decoder(self):
        stacked_decoder = keras.models.Sequential([
            keras.layers.Dense(2000, activation="selu", input_shape=[10], kernel_initializer=keras.initializers.lecun_normal()),
            keras.layers.Dense(500, activation="selu", kernel_initializer=keras.initializers.lecun_normal()),
            keras.layers.Dense(500, activation="selu", kernel_initializer=keras.initializers.lecun_normal()),
            keras.layers.Dense(136, activation="sigmoid", kernel_initializer=keras.initializers.lecun_normal()), ])
        return stacked_decoder

    def train(self, loss_funciton=keras.losses.binary_crossentropy, optimizer_=keras.optimizers.SGD(lr=1), number_of_epoch=10):
        stacked_ae = keras.models.Sequential([self._encoder(), self._decoder()])
        stacked_ae.compile(loss=loss_funciton,
                           optimizer=optimizer_,
                           metrics=['accuracy'])
        print('Training start')
        history = stacked_ae.fit(x=self.x_train, y=self.x_train, epochs=number_of_epoch,
                                 validation_data=[self.x_valid, self.x_valid])
        print('Training ended')
        return history


class DEC_greedy_autoencoder():

    def __init__(self, train, valid):
        self.x_train = train
        self.x_valid = valid
        self.autoencoder = None
        self.encoder = None

    def encode(self, data):
        if self.encoder is None:
            print("No encoder has been trained!")
            return None
        else:
            return self.encoder.predict(data)

    def predict(self, data):
        return self.model.predict(data)

    def greedy_pretraining(self, loss_function=keras.losses.binary_crossentropy, lr=0.1, finetune_epochs=100000, pretrain_epochs=50000, neuron_list=[500,500,2000,10], input_shape=136, dropout_rate=0.2, verbose=1):

        print(f'Adding and training first layer for {pretrain_epochs} epochs')

        neurons_first_layer = neuron_list.pop(0)

        # Create first Encoder decoder pair
        input = keras.layers.Input(shape=(input_shape,))
        dropout_out = keras.layers.Dropout(dropout_rate)(input)
        enc_layer = keras.layers.Dense(neurons_first_layer, activation='selu',input_shape=[input_shape], kernel_initializer=keras.initializers.lecun_normal())
        enc_out = enc_layer(dropout_out)
        dec_layer = keras.layers.Dense(input_shape, activation='selu',input_shape=[neurons_first_layer], kernel_initializer=keras.initializers.lecun_normal())
        dec_out = dec_layer(enc_out)

        model = keras.models.Model(inputs=input, outputs=dec_out)

        model.compile(loss=loss_function, optimizer=keras.optimizers.SGD(lr), metrics=['accuracy'])

        model.fit(x=self.x_train, y=self.x_train, epochs=pretrain_epochs, validation_data=[self.x_valid, self.x_valid], verbose=verbose)

        decoder_layer_list = [dec_layer]

        enc_out = enc_layer(input)
        trained_encoder = keras.models.Model(inputs=input, outputs=enc_out)

        # add and train more layers
        full_model = self.add_and_fit_layers(loss_function, lr, pretrain_epochs, decoder_layer_list, neuron_list, dropout_rate, trained_encoder, input, verbose)

        #finetune_model and return history

        print(f'Finetuning final model for {finetune_epochs} epochs')

        history = full_model.fit(x=self.x_train, y=self.x_train, epochs=finetune_epochs, validation_data=[self.x_valid, self.x_valid], batch_size=256, verbose=verbose)


        #Saving full autoencoder because why not?
        self.autoencoder = keras.models.clone_model(full_model)
        self.autoencoder.set_weights(full_model.get_weights())
        self.autoencoder.compile(loss=loss_function, optimizer=keras.optimizers.SGD(lr), metrics=['accuracy'])

        #extract encoder from autoencoder
        out = full_model.layers[1](input)
        full_encoder = keras.models.Model(inputs=input, outputs=out)
        full_encoder.compile(loss=loss_function, optimizer=keras.optimizers.SGD(lr), metrics=['accuracy'])

        self.encoder = full_encoder

        return history

    def add_and_fit_layers(self, loss_function, lr, pretrain_epochs, decoder_layer_list, neuron_list, dropout_rate, current_encoder_stack, input, verbose):

        which_layer = len(neuron_list)

        if which_layer == 0:
            return self.combine_encoder_decoder(current_encoder_stack, decoder_layer_list, input, loss_function, lr)

        print(f'Adding and training another layer for {pretrain_epochs} epochs')

        # encode data using already trained encoders. Creates data used for training following layers
        encoded_data_train = current_encoder_stack.predict(self.x_train)
        encoded_data_valid = current_encoder_stack.predict(self.x_valid)
        encoded_data_length = len(encoded_data_train[0])

        #Build new encoder + decoder pair
        n_neurons = neuron_list.pop(0)

        input_new_layer = keras.layers.Input(shape=(encoded_data_length,))

        dropout_out = keras.layers.Dropout(dropout_rate)(input_new_layer)

        #last encoder layer should not have activation function. allowing full expressiveness
        if which_layer == 1:
            new_enc_layer = keras.layers.Dense(n_neurons, input_shape=[encoded_data_length],
                                               kernel_initializer=keras.initializers.lecun_normal())
            new_enc_out = new_enc_layer(dropout_out)
        else:
            new_enc_layer = keras.layers.Dense(n_neurons, activation='selu', input_shape=[encoded_data_length],
                                               kernel_initializer=keras.initializers.lecun_normal())
            new_enc_out = new_enc_layer(dropout_out)

        #new_enc_layer = keras.layers.Dense(n_neurons, activation='selu', input_shape=[encoded_data_length], kernel_initializer=keras.initializers.lecun_normal())
        #new_enc_out = new_enc_layer(dropout_out)

        new_dec_layer = keras.layers.Dense(encoded_data_length, activation='selu', input_shape=[n_neurons], kernel_initializer=keras.initializers.lecun_normal())
        new_dec_out = new_dec_layer(new_enc_out)

        model = keras.models.Model(inputs=input_new_layer, outputs=new_dec_out)

        model.compile(loss=loss_function, optimizer=keras.optimizers.SGD(lr), metrics=['accuracy'])

        model.fit(x=encoded_data_train, y=encoded_data_train, epochs=pretrain_epochs, validation_data=[encoded_data_valid, encoded_data_valid], verbose=verbose)

        #puts in the opposite end of append()
        decoder_layer_list.insert(0, new_dec_layer)

        #add new encoder layer to previous encoder layers
        old_out = current_encoder_stack(inputs=input)
        out = new_enc_layer(old_out)

        model = keras.models.Model(inputs=input, outputs=out)
        #skal man compile her?
        model.compile(loss=loss_function, optimizer=keras.optimizers.SGD(lr), metrics=['accuracy'])

        return self.add_and_fit_layers(loss_function, lr, pretrain_epochs, decoder_layer_list, neuron_list, dropout_rate, model, input, verbose)

    def combine_encoder_decoder(self, encoder, decoder_layers, input, loss_function, lr):

        out = encoder(input)
        for layer in decoder_layers:
            out = layer(out)

        model = keras.models.Model(inputs=input, outputs=out)
        # skal man compile her?
        model.compile(loss=loss_function, optimizer=keras.optimizers.SGD(lr), metrics=['accuracy'])

        return model

    def sådan_virker_pretraining(self, loss_function=keras.losses.binary_crossentropy, lr=1, number_of_epochs=10, input_shape=136):

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

        model2.fit(x=self.x_train, y=self.x_train, epochs=number_of_epochs, validation_data=[self.x_valid, self.x_valid])

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


