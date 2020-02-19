from pickle import load
from random import sample
import numpy as np
from tensorflow.keras.utils import to_categorical

from utils import *


def load_data(data_train_model: DataSpec, data_test_model: DataSpec, data_train_monitor: DataSpec,
              data_test_monitor: DataSpec, data_run: DataSpec,
              pixel_depth=None):
    data_specs = [data_train_model, data_test_model, data_train_monitor, data_test_monitor, data_run]
    data_specs_network = [data_train_model, data_test_model]
    data_specs_rest = [data_train_monitor, data_test_monitor, data_run]

    # load data from files and pickle it
    for ds in data_specs:
        if not ds.has_data():
            if ds.file is None:
                raise(ValueError("Got a DataSpec with neither data nor a file name specified!"))

            file = "../" + ds.file  # go up one folder because run scripts are started from the folder "run/"
            with open(file, mode='rb') as f:
                data = load(f)
                ds.set_data(data=data)
            assert ds.has_data(), "Was not able to find data!"

    # filter data such that only the specified classes occur
    for ds in data_specs:
        if ds.classes is not None:
            x_new = []
            y_new = []
            for x, y in zip(ds.x(), ds.y()):
                if y in ds.classes:
                    x_new.append(x)
                    y_new.append(y)
            ds.set_data(x=np.array(x_new), y=np.array(y_new))

    # account for the correct number of data points
    for ds in data_specs:
        length = len(ds.y())
        if ds.n is None or ds.n > length:
            ds.n = length
        elif ds.n < length:
            if ds.randomize:
                # sample uniformly from the whole data
                indices = sample(range(length), ds.n)
                x = ds.x()[indices]
                y = ds.y()[indices]
            else:
                # choose the first n data points
                x = ds.x()[:ds.n]
                y = ds.y()[:ds.n]
            ds.set_data(x=x, y=y)

    # normalize data by pixel depth
    if pixel_depth is not None:
        for ds in data_specs:
            x = (ds.x().astype(np.float32) - (pixel_depth * 0.5)) / (pixel_depth * 0.5)
            ds.set_data(x=x, y=ds.y())

    # normalize labels to "categorical vector"
    all_classes_network = get_labels(data_specs_network)
    all_classes_rest = get_labels(data_specs_rest)

    return all_classes_network, all_classes_rest


def get_labels(data_specs):
    all_classes_total = set()

    for ds in data_specs:
        all_classes_data = set()
        if len(ds.y().shape) == 1:
            # collect all classes
            for yi in ds.y():
                all_classes_data.add(yi)
        else:
            # collect all classes
            for yi in ds.y():
                all_classes_data.add(np.argmax(yi))

        if len(ds.y().shape) == 1:
            # normalize labels to "categorical vector"
            ds.set_y(to_categorical(ds.y(), num_classes=number_of_classes(all_classes_data), dtype='float32'))
        for cd in all_classes_data:
            all_classes_total.add(cd)

    return sorted(all_classes_total)
