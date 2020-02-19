from . import *


def get_model_loader(name: str, model_path):
    if name == "GTSRB":
        model_constructor = GTSRB_CNY19
    elif name == "MNIST":
        model_constructor = MNIST_CNY19
    elif name == "F_MNIST":
        model_constructor = F_MNIST_CNY19
    elif name == "RESNET":
        model_constructor = ResNet50_19
    elif name == "VGG16":
        model_constructor = VGG16_19
    elif name == "ToyModel":
        model_constructor = ToyModel
    elif name == "VGG_CIFAR10":
        model_constructor = VGG_CIFAR10
    elif name == "CIFAR":
        model_constructor = CIFAR_CNY19
    else:
        raise(ValueError("Could not find model " + name + "!"))

    # prepend "models" folder
    model_path = "models/" + model_path

    return model_path, model_constructor
