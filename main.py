import math
import autoencoder_simple
import clustering_k_means
import vamb_tools
import matplotlib.pyplot as plt
import datetime

if __name__ == '__main__':

    print(datetime.datetime.now())

    with vamb_tools.Reader('C:/Users/M0107/Desktop/p10/Bin.gz', 'rb') as filehandle:
        tnfs, contigname, lengths = vamb_tools.read_contigs(filehandle, minlength=4)

    print(datetime.datetime.now())

    split_length = math.floor(len(tnfs) * 0.8)

    train = tnfs[:split_length, :]
    val = tnfs[split_length + 1:, :]

    AE = autoencoder_simple.stacked_autoencoder(train=train, valid=val)
    history = AE.train(number_of_epoch=10)


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

    print('end')