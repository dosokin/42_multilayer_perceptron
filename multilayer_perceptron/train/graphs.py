import numpy as np
from matplotlib import pyplot as plt


def display_graphs(accuracy, train_loss, val_loss, epoch_count):

    plt.subplot(2, 1, 1)

    plt.plot(np.arange(1, epoch_count + 1, 1), np.array(train_loss), 'r')
    plt.plot(np.arange(1, epoch_count + 1, 1), np.array(val_loss), 'b--')
    plt.legend(['Training loss', 'Validation loss'])
    plt.xlabel = "Epochs"

    plt.subplot(2, 1, 2)

    plt.plot(np.arange(1, epoch_count + 1, 1), np.array(accuracy), color='g', label="Accuracy")
    plt.legend(['Accuracy'])
    plt.xlabel = "Epochs"

    plt.show()
