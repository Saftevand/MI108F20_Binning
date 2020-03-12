import math
import autoencoder_simple
import clustering_k_means
import data_processor as _dp
import vamb_tools
import matplotlib.pyplot as plt
import datetime
import multiprocessing as _multiprocessing
import numpy as np
from tensorflow import keras

if __name__ == '__main__':
    _multiprocessing.freeze_support() # Skal være her så længe at vi bruger vambs metode til at finde depth


    #tnfs = _dp.get_tnfs()
    #np.save('tnfs.npy', tnfs)
    tnfs = np.load('tnfs.npy')

    split_length = math.floor(len(tnfs) * 0.8)
    train = tnfs[:split_length, :]
    val = tnfs[split_length + 1:, :]

    AE = autoencoder_simple.stacked_autoencoder(train=train, valid=val)
    history = AE.train(number_of_epoch=5000, loss_funciton=keras.losses.mean_absolute_error)



    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
