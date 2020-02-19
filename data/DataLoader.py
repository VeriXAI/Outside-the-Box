from . import *


def get_data_loader(string):
    if string == "GTSRB":
        return load_GTSRB
    elif string == "CIFAR10":
        return load_CIFAR_10
    elif string == "MNIST":
        return load_MNIST
    elif string == "F_MNIST":
        return load_F_MNIST
    elif string == "ToyData":
        return load_ToyData
    else:
        raise(ValueError("Could not find data " + string + "!"))
