from utils import *
from monitoring import *
from abstractions import *


def load_instance(n_classes, total_classes, stored_network_name):
    Monitor.reset_ids()
    classes = [k for k in range(n_classes)]
    data_train_model = DataSpec(randomize=False, classes=classes)
    data_test_model = DataSpec(randomize=False, classes=classes)
    data_train_monitor = DataSpec(randomize=False, classes=classes)
    data_test_monitor = DataSpec(randomize=False, classes=classes)
    data_run = DataSpec(randomize=False, classes=[k for k in range(0, total_classes)])
    classes_string = classes2string(classes)
    model_path = "{}_{}.h5".format(stored_network_name, classes_string)

    return data_train_model, data_test_model, data_train_monitor, data_test_monitor, data_run, model_path, \
        classes_string


def print_general_statistics(statistics, data_train_monitor, data_run):
    print("overall statistics")
    print("{} seconds for extracting {:d} values during monitor training".format(
        float_printer(statistics.time_training_monitor_value_extraction), data_train_monitor.n))
    print("{} seconds for clustering during monitor training".format(
        float_printer(statistics.time_training_monitor_clustering)))
    print("{} seconds for monitor training on {:d} samples".format(
        float_printer(statistics.time_training_monitor_tweaking), data_train_monitor.n))
    print("{} seconds for extracting {:d} values during running the monitored session".format(
        float_printer(statistics.time_running_monitor_value_extraction), data_run.n))
    print("{} seconds for running the monitored session on {:d} samples".format(
        float_printer(statistics.time_running_monitor_classification), data_run.n))


def print_monitor_statistics_single(monitor, statistics, data_train_monitor, data_run):
    m_id = monitor.id()
    print("\nprinting statistics for monitor {:d} with abstraction structure {}".format(m_id, monitor.short_str()))
    time_training = statistics.time_tweaking_each_monitor[m_id]
    print("{} seconds for training the monitor on {:d} samples".format(
        float_printer(time_training), data_train_monitor.n))
    time_running = statistics.time_running_each_monitor[m_id]
    print("{} seconds for running the monitor on {:d} samples".format(
        float_printer(time_running), data_run.n))
    return time_training, time_running


def print_monitor_statistics(monitors, statistics, data_train_monitor, data_run):
    for monitor in monitors:
        print_monitor_statistics_single(monitor, statistics, data_train_monitor, data_run)


# modifies: storage_monitors
def print_and_store_monitor_statistics(storage_monitors, monitors, statistics, history_run, novelty_wrapper_run,
                                       data_train_monitor, data_run):
    for monitor in monitors:
        m_id = monitor.id()
        time_training, time_running = print_monitor_statistics_single(monitor, statistics, data_train_monitor, data_run)
        history_run.update_statistics(m_id)
        fn = history_run.false_negatives()
        fp = history_run.false_positives()
        tp = history_run.true_positives()
        tn = history_run.true_negatives()
        novelty_results = novelty_wrapper_run.evaluate_detection(m_id)
        storage = CoreStatistics(fn=fn, fp=fp, tp=tp, tn=tn,
                                 novelties_detected=len(novelty_results["detected"]),
                                 novelties_undetected=len(novelty_results["undetected"]),
                                 time_training=time_training, time_running=time_running)
        storage_monitors[m_id - 1].append(storage)


def instance_MNIST():
    model_name = "MNIST"
    data_name = "MNIST"
    stored_network_name = "CNY19id1_MNIST"
    total_classes = 10
    return model_name, data_name, stored_network_name, total_classes


def instance_F_MNIST():
    model_name = "F_MNIST"
    data_name = "F_MNIST"
    stored_network_name = "CNY19id1_F_MNIST"
    total_classes = 10
    return model_name, data_name, stored_network_name, total_classes


def instance_CIFAR10():
    model_name = "CIFAR"
    data_name = "CIFAR10"
    stored_network_name = "CNY19id2_CIFAR"
    total_classes = 10
    return model_name, data_name, stored_network_name, total_classes


def instance_GTSRB():
    model_name = "GTSRB"
    data_name = "GTSRB"
    stored_network_name = "CNY19id2_GTSRB"
    total_classes = 43
    return model_name, data_name, stored_network_name, total_classes


def box_abstraction_given_layers(layers):
    layer2abstraction = dict()
    for layer in layers:
        layer2abstraction[layer] = BoxAbstraction(euclidean_distance)
    return layer2abstraction


def box_abstraction_MNIST(learn_from_test_data=False):
    layer2abstraction = box_abstraction_given_layers([-1, -2, -3, -4])
    return Monitor(layer2abstraction=layer2abstraction, learn_from_test_data=learn_from_test_data)


def box_abstraction_F_MNIST(learn_from_test_data=False):
    layer2abstraction = box_abstraction_given_layers([-1, -2, -3, -4, -5])
    return Monitor(layer2abstraction=layer2abstraction, learn_from_test_data=learn_from_test_data)


def box_abstraction_CIFAR10(learn_from_test_data=False):
    layer2abstraction = box_abstraction_given_layers([-1, -2, -3, -4])
    return Monitor(layer2abstraction=layer2abstraction, learn_from_test_data=learn_from_test_data)


def box_abstraction_GTSRB(learn_from_test_data=False):
    layer2abstraction = box_abstraction_given_layers([-2])
    return Monitor(layer2abstraction=layer2abstraction, learn_from_test_data=learn_from_test_data)
