from run.experiment_helper import *
from run.Runner import run


def run_experiment_distance():
    # global options
    seed = 0
    confidence_thresholds = uniform_bins(50, max=0.5)
    logger = Logger.start("log_distance.txt")

    # instance options
    model_name, data_name, stored_network_name, total_classes = instance_MNIST()
    title = "Decision performance distance_MNIST"
    clustering_threshold = 0.07
    n_classes = 5

    # load instance
    data_train_model, data_test_model, data_train_monitor, data_test_monitor, data_run, model_path, classes_string = \
        load_instance(n_classes, total_classes, stored_network_name)

    # create monitor
    monitors = [box_abstraction_MNIST()]
    monitor_manager = MonitorManager(monitors, clustering_threshold=clustering_threshold, skip_confidence=False)

    # run instance
    history_run, novelty_wrapper_run, statistics = evaluate_all(
        seed=seed, data_name=data_name, data_train_model=data_train_model, data_test_model=data_test_model,
        data_train_monitor=data_train_monitor, data_test_monitor=data_test_monitor, data_run=data_run,
        model_name=model_name, model_path=model_path, monitor_manager=monitor_manager, n_epochs=None,
        batch_size=None, model_trainer=None)

    # print statistics
    print_general_statistics(statistics, data_train_monitor, data_run)
    print_monitor_statistics(monitors, statistics, data_train_monitor, data_run)

    # plot results
    for monitor in monitors:
        m_id = monitor.id()
        plot_false_decisions([m_id], history_run, confidence_thresholds=confidence_thresholds,
                             name="{:d}, {}".format(m_id, classes_string), title=title)
    save_all_figures(close=True)

    # close log
    logger.stop()


if __name__ == "__main__":
    run_experiment_distance()
