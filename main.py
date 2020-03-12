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
from sklearn.preprocessing import normalize

if __name__ == '__main__':
    _multiprocessing.freeze_support() # Skal være her så længe at vi bruger vambs metode til at finde depth

    # vamb_depth = _dp.get_depth()
    # vamb_tnfs = _dp.get_tnfs('E:/Repositories/MI108F20_Binning/test/bigfasta.fna.gz')
    # np.save('vamb_tnfs', vamb_tnfs)
    # np.save('vamb_depths', vamb_depth)
    # tnfs = _dp.get_tnfs()
    # np.save('tnfs.npy', tnfs)
    tnfs = np.load('vamb_tnfs.npy')
    depth = np.load('vamb_depths.npy')
    # depth = np.load('depth.pny')
    # tnfs = np.load('tnfs.npy')


    norm_depth = normalize(depth, axis=1, norm='l1')
    input_array = np.hstack([tnfs, norm_depth])
    split_length = math.floor(len(input_array) * 0.8)
    train = input_array[:split_length, :]
    val = input_array[split_length + 1:, :]

    AE = autoencoder_simple.stacked_autoencoder(train=train, valid=val)
    history = AE.train(number_of_epoch=500, loss_funciton=keras.losses.mean_absolute_error)



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
