import tensorflow as tf
from tensorflow import keras


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


