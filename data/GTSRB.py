from utils import DataSpec, load_data, filter_labels


def load_GTSRB(data_train_model: DataSpec, data_test_model: DataSpec, data_train_monitor: DataSpec,
               data_test_monitor: DataSpec, data_run: DataSpec):
    # names of the data files
    data_train_model.file = "data/GTSRB/train.p"
    data_test_model.file = "data/GTSRB/test.p"
    data_train_monitor.file = data_train_model.file  # use training data for training
    data_test_monitor.file = data_test_model.file  # use testing data for running
    data_run.file = data_test_model.file  # use testing data for running

    pixel_depth = 255.0

    all_classes_network, all_classes_rest = load_data(
        data_train_model=data_train_model, data_test_model=data_test_model, data_train_monitor=data_train_monitor,
        data_test_monitor=data_test_monitor, data_run=data_run, pixel_depth=pixel_depth)

    # labels
    labels_all = ['label' + str(i) for i in range(43)]  # dummy names, TODO add correct names
    labels_all[0] = "20 km/h"
    labels_all[1] = "30 km/h"
    labels_all[2] = "50 km/h"
    labels_all[10] = "no passing"

    labels_network = filter_labels(labels_all, all_classes_network)
    labels_rest = filter_labels(labels_all, all_classes_rest)

    return all_classes_network, labels_network, all_classes_rest, labels_rest
