from utils import *
from trainers import *


def run_script():
    seed = 0
    data_name = "MNIST"
    model_name = "MNIST"
    n_epochs = 10
    batch_size = 128
    plot_name = "!"  # None = no plots, "" = show plots, "!" = store plots

    for n_classes in range(2, 9+1):
        classes = [k for k in range(n_classes)]
        data_train_model = DataSpec(randomize=False, classes=classes)
        data_test_model = DataSpec(randomize=False, classes=classes)
        classes_string = classes2string(classes)
        model_path = "CNY19id1_MNIST_{}.h5".format(classes_string)
        if plot_name == "!":
            plot_name_current = classes_string
        else:
            plot_name_current = plot_name

        run_training(
            data_name=data_name, data_train_model=data_train_model, data_test_model=data_test_model,
            model_name=model_name, model_path=model_path, n_epochs=n_epochs, batch_size=batch_size,
            seed=seed, plot_name=plot_name_current)


if __name__ == "__main__":
    run_script()
