import math
import autoencoder_simple
import data_processor as _dp
import matplotlib.pyplot as plt
import multiprocessing as _multiprocessing
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import normalize
import seaborn as sns
from tsnecuda import TSNE

if __name__ == '__main__':
    _multiprocessing.freeze_support() # Skal være her så længe at vi bruger vambs metode til at finde depth

    use_depth = True
    load_data = True
    #simon_stationær_path = 'C:/Users/Simon/Documents/GitHub/MI108F20_Binning/test/Bin.gz'

    if(use_depth):
        if(load_data):
            tnfs = np.load('vamb_tnfs.npy')
            depth = np.load('vamb_depths.npy')
        else:
            tnfs = _dp.get_tnfs()
            depth = _dp.get_depth()
            np.save('vamb_depths.npy', depth)
            np.save('vamb_tnfs.npy', tnfs)
        norm_depth = normalize(depth, axis=1, norm='l1')
        input_array = np.hstack([tnfs, norm_depth])
        split_length = math.floor(len(input_array) * 0.8)
        train = input_array[:split_length, :]
        val = input_array[split_length + 1:, :]
    else:
        if (load_data):
            tnfs = np.load('vamb_tnfs.npy')
        else:
            tnfs = _dp.get_tnfs()
        split_length = math.floor(len(tnfs) * 0.8)
        train = tnfs[:split_length, :]
        val = tnfs[split_length + 1:, :]
    #AE = autoencoder_simple.stacked_autoencoder(train=train, valid=val, input_layer_size=len(train[0]))
    #history, model = AE.train(number_of_epoch=1)

    #test = AE.encoder.predict(tnfs)
    #k_means = clustering_k_means.clustering_k_means(k_clusters=5)
    #k_means.do_clustering(dataset=test, max_iterations=5)



    #2000 og 5000 virker ret fint umiddelbart - lille dataset
    AE = autoencoder_simple.DEC_greedy_autoencoder(train=train, valid=val)
    history = AE.greedy_pretraining(loss_function=keras.losses.mean_absolute_error, pretrain_epochs=2000,
                                    finetune_epochs=5000, lr=0.01, neuron_list=[500, 500, 2000, 10],
                                    input_shape=len(train[0]), dropout_rate=0.2, verbose=0)

    encoded_train = AE.encode(train)

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


    ''' Ting og sager til TSNE --> virker kun til linux (men virker det?)
    X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(encoded_train)

    plt.figure(figsize=(16, 10))
    sns.scatterplot(hue='y',
                    palette=sns.color_palette("hls", 10),
                    data=X_embedded,
                    legend="full",
                    alpha=0.3)
    '''