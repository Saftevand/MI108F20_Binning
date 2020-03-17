import autoencoders
import clustering_methods
import multiprocessing as _multiprocessing
from tensorflow import keras
import data_processor
import Visualizer
import binner




if __name__ == '__main__':
    _multiprocessing.freeze_support()  # Skal være her så længe at vi bruger vambs metode til at finde depth

    #dp = data_processor.Data_processor()

    #train, val = dp.get_train_and_validation_data()

    #AE = autoencoders.Stacked_autoencoder(train=train, valid=val, input_layer_size=len(train[0]))
    #history, model = AE.train(number_of_epoch=5)

    #test = AE.encoder.predict(train)
    #k_means = clustering_methods.clustering_k_means(k_clusters=5)


    binner_ = binner.Binner(autoencoder=autoencoders.DEC_greedy_autoencoder(train=None, valid=None), clustering_method= clustering_methods.clustering_k_means())

    binner_.extract_features()
    binner_.do_clustering()

    print("breakpoint")




    #2000 og 5000 virker ret fint umiddelbart - lille dataset
    #AE = autoencoders.DEC_greedy_autoencoder()
    #history = AE.greedy_pretraining(loss_function=keras.losses.mean_absolute_error, pretrain_epochs=2000,
    #                                finetune_epochs=5000, lr=0.01, neuron_list=[500, 500, 2000, 10], dropout_rate=0.2, verbose=0)

    #encoded_train = AE.encode(train)

    #Visualizer.training_graphs(history=history)


    #ref1 = train[0]
    #res = AE.predict(train)

    ''' Ting og sager til TSNE --> virker kun til linux (men virker det?)
    X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(encoded_train)

    plt.figure(figsize=(16, 10))
    sns.scatterplot(hue='y',
                    palette=sns.color_palette("hls", 10),
                    data=X_embedded,
                    legend="full",
                    alpha=0.3)
    '''