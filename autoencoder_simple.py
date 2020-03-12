import tensorflow as tf
from tensorflow import keras



class stacked_autoencoder():

    def __init__(self, train, valid):
        self.x_train = train
        self.x_valid = valid

    def _encoder(self):
        stacked_encoder = keras.models.Sequential([
            keras.layers.Dense(68, activation="selu", input_shape=[136,], kernel_initializer=keras.initializers.lecun_normal()),
            keras.layers.Dense(34, activation="selu", kernel_initializer=keras.initializers.lecun_normal()), ])
        return stacked_encoder

    def _decoder(self):
        stacked_decoder = keras.models.Sequential([
            keras.layers.Dense(68, activation="selu", input_shape=[34], kernel_initializer=keras.initializers.lecun_normal()),
            keras.layers.Dense(136, activation="sigmoid", kernel_initializer=keras.initializers.RandomNormal), ])
        return stacked_decoder

    def train(self, loss_funciton=keras.losses.binary_crossentropy, optimizer_=keras.optimizers.SGD(lr=1.5), number_of_epoch=10):
        stacked_ae = keras.models.Sequential([self._encoder(), self._decoder()])
        stacked_ae.compile(loss= loss_funciton, #"binary_crossentropy",
                           optimizer= optimizer_,
                           metrics=['accuracy'])
        print('Training start')
        history = stacked_ae.fit(x=self.x_train, y=self.x_train, epochs=number_of_epoch,
                                 validation_data=[self.x_valid, self.x_valid])
        print('Training ended')
        return history


