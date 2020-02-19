import numpy as np

from utils import DataSpec, load_data


def load_ToyData(data_train_model: DataSpec, data_test_model: DataSpec, data_train_monitor: DataSpec,
                 data_test_monitor: DataSpec, data_run: DataSpec):
    # add data
    x = np.array([[0.7, 0.2], [0.6, 0.2], [0.7, 0.1], [0.8, 0.1],  # class 1, first cluster
                  [0.9, 0.2],  # class 1, second cluster
                  [0.5, 0.5], [0.5, 0.6], [0.4, 0.6],  # class 2, first cluster
                  [0.2, 0.7]  # class 2, second cluster
                  ])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])
    data_train_model.set_data(x=x, y=y)
    data_test_model.set_data(x=x, y=y)
    data_train_monitor.set_data(x=x, y=y)
    data_test_monitor.set_data(x=x, y=y)
    data_run.set_data(x=x, y=y)

    pixel_depth = None

    load_data(data_train_model=data_train_model, data_test_model=data_test_model, data_train_monitor=data_train_monitor,
              data_test_monitor=data_test_monitor, data_run=data_run, pixel_depth=pixel_depth)

    # labels
    all_classes_network = [0, 1]
    labels_network = ["label1", "label2"]
    all_classes_rest = all_classes_network
    labels_rest = labels_network

    return all_classes_network, labels_network, all_classes_rest, labels_rest
