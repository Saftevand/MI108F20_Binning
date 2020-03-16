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

    #simon_stationær_path = 'C:/Users/Simon/Documents/GitHub/MI108F20_Binning/test/Bin.gz'

    #tnfs = _dp.get_tnfs(simon_stationær_path)
    #np.save('tnfs.npy', tnfs)
    tnfs = np.load('tnfs.npy')

    split_length = math.floor(len(tnfs) * 0.8)
    train = tnfs[:split_length, :]
    val = tnfs[split_length + 1:, :]

    #AE = autoencoder_simple.stacked_autoencoder(train=train, valid=val)
    #AE = autoencoder_simple.DEC_autoencoder(train=train, valid=val)

    AE = autoencoder_simple.DEC_greedy_autoencoder(train=train, valid=val)
    history = AE.greedy_pretraining(loss_function=keras.losses.mean_absolute_error, pretrain_epochs=50, finetune_epochs=200, lr=0.1, neuron_list=[500, 500, 2000, 10], input_shape=136, dropout_rate=0.2)

    #history = AE.train(number_of_epoch=500, loss_funciton=keras.losses.mean_absolute_error)



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

    ref1 = train[0]
    res = AE.predict(train)

    print(f'correct: {ref1[0]}, \n predict: {res[0]} \n \n')

    ref2 = [train[1]]
    print(f'correct: {ref2[0]}, \n predict: {res[1]}')
