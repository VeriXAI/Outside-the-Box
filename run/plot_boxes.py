from data import *
from run.experiment_helper import *


def run_script():
    model_name, data_name, stored_network_name, total_classes = instance_MNIST()
    classes = [0, 1]
    n_classes = 2
    classes_string = classes2string(classes)
    model_path = "{}_{}.h5".format(stored_network_name, classes_string)
    data_train_model = DataSpec(randomize=False, classes=classes)
    data_test_model = DataSpec(randomize=False, classes=classes)
    data_train_monitor = DataSpec(randomize=False, classes=classes)
    data_test_monitor = DataSpec(randomize=False, classes=classes)
    data_run = DataSpec(randomize=False, classes=[0, 1, 2])

    all_classes_network, labels_network, all_classes_rest, labels_rest = get_data_loader(data_name)(
        data_train_model=data_train_model, data_test_model=data_test_model, data_train_monitor=data_train_monitor,
        data_test_monitor=data_test_monitor, data_run=data_run)
    model, _ = get_model(model_name=model_name, data_train=data_train_model, data_test=data_test_model,
                         n_classes=n_classes, model_trainer=None, n_epochs=None, batch_size=None, statistics=None,
                         model_path=model_path)

    # create monitor
    layer2abstraction = {-2: BoxAbstraction(euclidean_distance)}
    monitor = Monitor(layer2abstraction=layer2abstraction)
    monitor_manager = MonitorManager([monitor], n_clusters=3)

    # run instance
    monitor_manager.normalize_and_initialize(model, len(labels_rest))
    monitor_manager.train(model=model, data_train=data_train_monitor, data_test=data_test_monitor,
                          statistics=Statistics())

    # create plot
    history = History()
    history.set_ground_truths(data_run.ground_truths())
    layer = 8
    layer2values, _ = obtain_predictions(model=model, data=data_run, layers=[layer])
    history.set_layer2values(layer2values)
    plot_2d_projection(history=history, monitor=monitor, layer=layer, category_title=model_name, known_classes=classes,
                       novelty_marker="*", dimensions=[3, 13])

    save_all_figures()


if __name__ == "__main__":
    run_script()
