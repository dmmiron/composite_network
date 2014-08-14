import sys
import matplotlib.pyplot as plt

from pylearn2.utils import serial


def load(model_files):
    models = []
    for model_file in model_files:
        models.append(serial.load(model_file))
    return models

def load_from_file(file_name):
    f = open(file_name, "rb")
    model_files = []
    for line in f:
        model_files.append(line.strip("\n"))
    return load(model_files), model_files

def plot(models, model_files):
    for model, model_file in zip(models, model_files):
        channel = model.monitor.channels['valid_y_misclass']
        y = channel.val_record
        seconds = channel.time_record
        epochs = channel.epoch_record
        plt.subplot(211)
        plt.xlabel("seconds")
        plt.ylabel("valid_y_misclass")
        plt.plot(seconds, y, label=model_file)
        plt.subplot(212)
        plt.xlabel("epochs")
        plt.ylabel("valid_y_misclass")
        plt.plot(epochs, y, label=model_file)
    plt.subplot(211)    
    plt.legend(loc='upper right')
    plt.subplots_adjust(left=0.125, right=1.9, bottom=0.1, top=1.9, hspace=0.5)
    plt.show()

if __name__ == "__main__":
    if (sys.argv[1] == "-f"):
        file_name = sys.argv[2]
        models, model_files = load_from_file(file_name)
    else:
       model_files = sys.argv[1:]
       models = load(model_files)

    plot(models, model_files)
