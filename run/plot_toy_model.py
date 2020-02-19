from data import *
from utils import *
from abstractions import *
from trainers import *
from monitoring import *


def run_script():
    # options
    seed = 0
    data_name = "ToyData"
    classes = [0, 1]
    n_classes = 2
    data_train_model = DataSpec(classes=classes)
    data_test_model = DataSpec(classes=classes)
    data_train_monitor = DataSpec(classes=classes)
    data_test_monitor = DataSpec(classes=classes)
    data_run = DataSpec(classes=classes)
    model_name = "ToyModel"
    model_path = "Toy-model.h5"
    n_epochs = 0
    batch_size = 128
    model_trainer = StandardTrainer()

    all_classes_network, labels_network, all_classes_rest, labels_rest = get_data_loader(data_name)(
        data_train_model=data_train_model, data_test_model=data_test_model, data_train_monitor=data_train_monitor,
        data_test_monitor=data_test_monitor, data_run=data_run)
    model, _ = get_model(model_name=model_name, data_train=data_train_model, data_test=data_test_model,
                         n_classes=n_classes, model_trainer=model_trainer, n_epochs=n_epochs, batch_size=batch_size,
                         statistics=Statistics(), model_path=model_path)

    # create monitor
    layer2abstraction = {1: BoxAbstraction(euclidean_distance)}
    monitor = Monitor(layer2abstraction=layer2abstraction)
    monitor_manager = MonitorManager([monitor], n_clusters=1)

    # run instance
    monitor_manager.normalize_and_initialize(model, len(labels_rest))
    monitor_manager.train(model=model, data_train=data_train_monitor, data_test=data_test_monitor,
                          statistics=Statistics(), ignore_misclassifications=False)

    # create plots
    history = History()
    history.set_ground_truths(data_run.ground_truths())
    layer = 1
    layer2values, _ = obtain_predictions(model=model, data=data_run, layers=[layer])
    history.set_layer2values(layer2values)
    plot_2d_projection(history=history, monitor=None, layer=layer, category_title=model_name, known_classes=classes,
                       novelty_marker="*", dimensions=[0, 1])
    plot_2d_projection(history=history, monitor=monitor, layer=layer, category_title=model_name, known_classes=classes,
                       novelty_marker="*", dimensions=[0, 1])

    save_all_figures(close=True)


if __name__ == "__main__":
    run_script()
