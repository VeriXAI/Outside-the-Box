from utils import *
from models import *
from data import *


def print_statistics():
    for i, model_constructor, data_loader, n_classes in [(1, MNIST_CNY19, load_MNIST, 10),
                                                         (2, GTSRB_CNY19, load_GTSRB, 43)]:
        classes = [k for k in range(n_classes)]
        data_train_model = DataSpec(randomize=False, classes=classes)
        data_test_model = DataSpec(randomize=False, classes=classes)
        data_train_monitor = DataSpec(randomize=False, classes=classes)
        data_test_monitor = DataSpec(randomize=False, classes=classes)
        data_run = DataSpec(randomize=False, classes=classes)
        data_loader(data_train_model=data_train_model, data_test_model=data_test_model,
                    data_train_monitor=data_train_monitor, data_test_monitor=data_test_monitor, data_run=data_run)
        image_shape = get_image_shape(data_train_model.x())
        model = model_constructor(weights=None, classes=n_classes, input_shape=image_shape)

        print("network {:d} has {:d} hidden layers with {:} neurons".format(i, number_of_hidden_layers(model),
                                                                            number_of_hidden_neurons(model)))


if __name__ == "__main__":
    print_statistics()
