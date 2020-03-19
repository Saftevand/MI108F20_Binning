import matplotlib.pyplot as plt



def visualize2D(data, colors=None):
    if colors is None:
        colors = ['bo', 'ro', 'go', 'mo', 'ko', 'co', 'mo', 'yo', 'bx', 'rx', 'gx', 'mx', 'kx', 'cx', 'mx', 'yx',
                  'bv', 'rv', 'gv', 'mv', 'kv', 'cv', 'mv', 'yv', 'bs', 'rs', 'gs', 'ms', 'ks', 'cs', 'ms', 'ys',
                  'bp', 'rp', 'gp', 'mp', 'kp', 'cp', 'mp', 'yp', 'b+', 'r+', 'g+', 'm+', 'k+', 'c+', 'm+', 'y+']
    for i in range(0, data.axes[0].stop):
        for j in range(0, data.axes[1].stop):
            if (data[j][i] is not None):
                plt.plot(data[j][i][0], data[j][i][1], colors[i])
    plt.show()


def training_graphs(history):
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