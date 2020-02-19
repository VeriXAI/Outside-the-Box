from itertools import product

from run.experiment_helper import *
from utils import *


def run_experiment_layer_variation():
    # global options
    seed = 0
    n_layers = 3  # number of layers to watch
    n_monitors = (2**n_layers - 1) * 2
    logger = Logger.start("log_layer_variation.txt")

    # instance options
    model_name, data_name, stored_network_name, total_classes = instance_MNIST()
    clustering_threshold = 0.07

    storage_monitors = [[] for _ in range(n_monitors)]
    for n_classes in range(2, total_classes):
        print("\n--- new instance ---\n")
        # load instance
        data_train_model, data_test_model, data_train_monitor, data_test_monitor, data_run, model_path, _ = \
            load_instance(n_classes, total_classes, stored_network_name)

        # create (fresh) monitors
        monitors = []
        for permutation in product([0, 1], repeat=n_layers):
            layer2abstraction = dict()
            for i in range(n_layers):
                if permutation[i]:
                    layer2abstraction[-1-i] = BoxAbstraction(euclidean_distance)
            if len(layer2abstraction) == 0:
                continue
            monitor = Monitor(layer2abstraction=layer2abstraction)
            monitors.append(monitor)
        monitor_manager = MonitorManager(monitors, clustering_threshold=clustering_threshold)

        # run instance
        history_run, novelty_wrapper_run, statistics = evaluate_all(
            seed=seed, data_name=data_name, data_train_model=data_train_model, data_test_model=data_test_model,
            data_train_monitor=data_train_monitor, data_test_monitor=data_test_monitor, data_run=data_run,
            model_name=model_name, model_path=model_path, monitor_manager=monitor_manager)

        # print/store statistics
        print_general_statistics(statistics, data_train_monitor, data_run)
        print_and_store_monitor_statistics(storage_monitors, monitors, statistics, history_run,
                                           novelty_wrapper_run, data_train_monitor, data_run)

    # store results
    filename_prefix = "layers_" + data_name
    for monitor in monitors:
        m_id = monitor.id()
        store_core_statistics(storage_monitors[m_id - 1], "monitor{:d}".format(m_id),
                              filename_prefix=filename_prefix)

    # close log
    logger.stop()


def plot_experiment_layer_variation():
    # instance options
    data_name = "MNIST"
    n_layers = 3
    layers2index = {"001": 0, "010": 1, "100": 2, "011": 3, "101": 4, "110": 5, "111": 6}
    n_ticks = 8

    filename_prefix = "layers_" + data_name
    m_id = 0
    storage_all_sorted = [0 for _ in range(len(layers2index.values()))]
    for permutation in product([0, 1], repeat=n_layers):
        layers = [0 for _ in range(n_layers)]
        no_layer = True
        for i in range(n_layers):
            if permutation[i]:
                no_layer = False
                layers[-1 - i] = 1
        if no_layer:
            continue
        layer_string = ""
        for l in layers:
            layer_string += str(l)

        # uncomment to print the mapping from layers to indices
        # print("monitor ID {:d}, layers {} -> index {}".format(m_id, layers, layers2index[layer_string]))

        # load data
        m_id += 1
        storage = load_core_statistics("monitor{:d}".format(m_id), filename_prefix=filename_prefix)
        storage_all_sorted[layers2index[layer_string]] = storage

    plot_false_decisions_given_all_lists(storage_all_sorted, n_ticks=n_ticks, name=filename_prefix)

    save_all_figures(close=True)


def run_experiment_layer_variation_all():
    run_experiment_layer_variation()
    plot_experiment_layer_variation()


if __name__ == "__main__":
    run_experiment_layer_variation_all()
