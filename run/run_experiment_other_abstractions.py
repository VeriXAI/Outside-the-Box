from run.experiment_helper import *
from run.Runner import run


def run_experiment_other_abstractions():
    # global options
    seed = 0
    logger = Logger.start("log_other_abstractions.txt")

    # instance options
    model_name, data_name, stored_network_name, total_classes = instance_MNIST()
    clustering_threshold = 0.07

    storage_monitors = [[], [], [], [], []]
    for n_classes in range(2, total_classes):
        print("\n--- new instance ---\n")
        # load instance
        data_train_model, data_test_model, data_train_monitor, data_test_monitor, data_run, model_path, _ = \
            load_instance(n_classes, total_classes, stored_network_name)

        # create (fresh) monitors
        monitor1 = box_abstraction_MNIST()
        layer2abstraction = {-2: BoxAbstraction(euclidean_distance)}
        monitor2 = Monitor(layer2abstraction=layer2abstraction)
        layer2abstraction = {-2: MeanBallAbstraction(euclidean_distance)}
        monitor3 = Monitor(layer2abstraction=layer2abstraction)
        layer2abstraction = {-2: OctagonAbstraction(euclidean_distance)}
        monitor4 = Monitor(layer2abstraction=layer2abstraction)
        monitors = [monitor1, monitor2, monitor3, monitor4]
        monitor_manager = MonitorManager(monitors, clustering_threshold=clustering_threshold)

        # run instance
        history_run, novelty_wrapper_run, statistics = \
            evaluate_all(seed=seed, data_name=data_name, data_train_model=data_train_model,
                         data_test_model=data_test_model, data_train_monitor=data_train_monitor,
                         data_test_monitor=data_test_monitor, data_run=data_run, model_name=model_name,
                         model_path=model_path, monitor_manager=monitor_manager, n_epochs=None, batch_size=None,
                         model_trainer=None)

        # print/store statistics
        print_general_statistics(statistics, data_train_monitor, data_run)
        print_and_store_monitor_statistics(storage_monitors, monitors, statistics, history_run,
                                           novelty_wrapper_run, data_train_monitor, data_run)

    # store results
    filename_prefix = "other_abstractions_" + data_name
    for monitor in monitors:
        m_id = monitor.id()
        store_core_statistics(storage_monitors[m_id - 1], "monitor{:d}".format(m_id),
                              filename_prefix=filename_prefix)

    # close log
    logger.stop()


def plot_experiment_other_abstractions():
    # instance options
    data_name = "MNIST"
    n_ticks = 8

    filename_prefix = "other_abstractions_" + data_name
    storage_1 = load_core_statistics("monitor1", filename_prefix=filename_prefix)
    storage_2 = load_core_statistics("monitor2", filename_prefix=filename_prefix)
    storage_3 = load_core_statistics("monitor3", filename_prefix=filename_prefix)
    storage_4 = load_core_statistics("monitor4", filename_prefix=filename_prefix)
    storage_all = [storage_3, storage_4, storage_2, storage_1]

    plot_false_decisions_given_all_lists(storage_all, n_ticks=n_ticks, name=filename_prefix)

    save_all_figures(close=True)


def run_experiment_other_abstractions_all():
    run_experiment_other_abstractions()
    plot_experiment_other_abstractions()


if __name__ == "__main__":
    run_experiment_other_abstractions_all()
