import autoencoders
import clustering_methods
import multiprocessing as _multiprocessing
from tensorflow import keras
import data_processor
import Visualizer




if __name__ == '__main__':
    _multiprocessing.freeze_support()  # Skal være her så længe at vi bruger vambs metode til at finde depth

    dp = data_processor.Data_processor()

    train, val = dp.get_train_and_validation_data()

    #AE = autoencoders.Stacked_autoencoder(train=train, valid=val, input_layer_size=len(train[0]))
    #history, model = AE.train(number_of_epoch=5)

    #test = AE.encoder.predict(train)
    k_means = clustering_methods.clustering_k_means(k_clusters=5)

    AE = autoencoders.DEC_greedy_autoencoder(train=train, valid=val)
    history = AE.greedy_pretraining(loss_function=keras.losses.mean_absolute_error, pretrain_epochs=50, finetune_epochs=200, lr=0.1, neuron_list=[500, 500, 2000, 10], input_shape=136, dropout_rate=0.2)
    test = AE.predict(train)
    k_means.do_clustering(dataset=test, max_iterations=5)

    data = k_means.clustered_data

    Visualizer.training_graphs(history=history)


    ref1 = train[0]
    res = AE.predict(train)

    print(f'correct: {ref1[0]}, \n predict: {res[0]} \n \n')

    ref2 = [train[1]]
    print(f'correct: {ref2[0]}, \n predict: {res[1]}')
