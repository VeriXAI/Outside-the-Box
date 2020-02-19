from math import sqrt
from time import time
from datetime import time as time_type
import numpy as np
import colorsys
import random
from copy import copy
import csv
from tensorflow_core.python.keras.models import Model
from tensorflow_core.python.data import Dataset

from .CoreStatistics import CoreStatistics


def to_classes(list_of_bit_vectors):
    return [to_class(b) for b in list_of_bit_vectors]


def to_class(bit_vector):
    return np.where(bit_vector == 1)[0][0]


def to_dataset(x_train, y_train, batch_size):
    return Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)


def ratio(part, total):
    try:
        return part / total * 100
    except ZeroDivisionError:
        return 100.0
    except TypeError:
        return "?"


def extend(strings):
    n = max(len(s) for s in strings)
    return (s.rjust(n) for s in strings)


def get_rgb_colors(n):
    if n == 2:
        # special options for binary case
        return [(0.33, 0.42, 0.2), (0.96, 0.52, 0.26)]  # orange/green
        # return [(1., 0., 0.), (0, .4, .6)]  # red/blue

    # based on https://stackoverflow.com/a/876872
    hsv_tuples = [(x * 1.0 / n, 0.8, 1) for x in range(n)]
    rgb_colors = [colorsys.hsv_to_rgb(*x) for x in hsv_tuples]
    rgb_colors_shuffled = []
    step = int(sqrt(n))
    i = 0
    while i < step:
        for j in range(step):
            rgb_colors_shuffled.append(rgb_colors[i + j * step])
        i += 1
    rgb_colors_shuffled.extend(rgb_colors[i+(step-1)*step:])
    return rgb_colors_shuffled


def get_markers(n_classes):
    all_markers = ["o", "s", "^", "*", "p", "X", "D", "2", ".", "<", ">", "v"]
    if n_classes > len(all_markers):
        markers = copy(all_markers)
        while n_classes > len(markers):
            markers.extend(markers)
    else:
        markers = all_markers
    return markers[:n_classes]


def set_random_seed(n):
    print("Setting random seed to", n)
    random.seed(n)
    np.random.seed(n)


def get_image_shape(images):
    return images.shape[1:4]


def categoricals2numbers(categorical_vectors):
    """convert categorical vectors to numbers"""
    return [categorical2number(categorical_vector) for categorical_vector in categorical_vectors]


def categorical2number(categorical_vector):
    """convert categorical vector to number"""
    return np.where(categorical_vector == 1)[0][0]


def number_of_classes(classes):
    m = max(classes) + 1
    l = len(classes)
    return max(m, l)


def number_of_model_classes(model):
    return model.layers[-1].output_shape[1]


def rate_fraction(num, den):
    if den == 0:
        return 0  # convention: return 0
    return num/den


def obtain_predictions(model, data, layers=None, ignore_misclassifications: bool = False):
    delta_t = time()
    if layers is None:
        values = model.predict(data.x())
        result = to_classifications(values)
    else:
        if ignore_misclassifications:
            # compare classes to ground truths
            classes, _ = obtain_predictions(model, data)
            filter = []
            for i, (p, gt) in enumerate(zip(classes, data.ground_truths())):
                if p == gt:
                    filter.append(i)
            data.filter(filter)
        layer2values = dict()
        for layer_index in layers:
            try:
                manual_model = model.is_manual_model()
            except:
                manual_model = False
            if manual_model:
                result = model.predict(data.x(), layer_index)
            else:
                # construct pruned model following
                # https://keras.io/getting-started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer
                model_until_layer = Model(inputs=model.input, outputs=model.layers[layer_index].output)
                result = model_until_layer.predict(data.x())
            layer2values[layer_index] = result
        result = layer2values
    timer = time() - delta_t

    return result, timer


def to_classifications(list_of_predictions):
    return [to_classification(p) for p in list_of_predictions]


def to_classification(prediction):
    return np.argmax(prediction)


def filter_labels(all_labels, all_classes):
    if len(all_classes) < len(all_labels):
        return [all_labels[i] for i in range(max(all_classes) + 1)]
    else:
        return all_labels


def normalize_layer(model, raw_layer):
    layer_index = None
    if isinstance(raw_layer, str):
        # search for layer in the model
        for idx, layer in enumerate(model.layers):
            if layer.name == raw_layer:
                layer_index = idx
                break
    elif isinstance(raw_layer, int):
        if raw_layer < 0:
            layer_index = len(model.layers) + raw_layer
            assert layer_index >= 0, "Negative layer indices should be such that their absolute value is smaller " + \
                                     "than the number of layers."
        else:
            layer_index = raw_layer
            assert layer_index < len(model.layers), "Layer index exceeds the number of layers."
    else:
        raise (ValueError("A layer needs to be a string or an integer, but got ", raw_layer))

    if layer_index is None:
        raise (ValueError("Could not find layer", raw_layer))

    return layer_index


def float_printer(timer):
    if isinstance(timer, time_type):
        f = timer.second + timer.microsecond / 1000000
    else:
        assert isinstance(timer, int) or isinstance(timer, float)
        f = timer
    if f < 1e-2:
        if f == 0:
            return "0.00"
        return "< 0.01"
    return "{:.2f}".format(f)


def uniform_bins(n: int, max=1.0):
    step = max / float(n)
    return [i * step for i in range(n + 1)]


def determine_zero_filters(values: dict, data, n_neurons, layer=None):
    class2nonzeros = dict()
    for class_id in data.classes:
        class2nonzeros[class_id] = [0 for _ in range(n_neurons)]
    for vj, gt in zip(values, data.ground_truths()):
        for i, vi in enumerate(vj):
            if vi > 0:
                class2nonzeros[gt][i] += 1
    # create mask of all dimensions with entry 'True' whenever there is at least one non-zero entry
    class2nonzero_mask = dict()
    for class_id, nonzeros in class2nonzeros.items():
        nonzero_mask = []
        n_zeros = 0
        for i, nzi in enumerate(nonzeros):
            if nzi > 0:
                nonzero_mask.append(True)
            else:
                nonzero_mask.append(False)
                n_zeros += 1
        class2nonzero_mask[class_id] = nonzero_mask
        if layer is not None:
            print("filtering zeros removes {:d}/{:d} dimensions from layer {:d} for class {:d}".format(
                n_zeros, n_neurons, layer, class_id))
    return class2nonzero_mask


def classes2string(classes):
    if classes == [k for k in range(len(classes))]:
        # short version for consecutive classes
        return "0-{:d}".format(len(classes) - 1)
    else:
        # long version with enumeration of all classes
        comma = ""
        string = ""
        for c in classes:
            string += comma + str(c)
            comma = ","
    return string


def store_core_statistics(storages, name, filename_prefix="results"):
    if isinstance(name, str):
        filename = "{}-{}.csv".format(filename_prefix, name)
        _store_core_statistics_helper(filename, storages)
    else:
        assert isinstance(name, list)
        for storages_alpha, alpha in zip(storages, name):
            filename = "{}-at{}.csv".format(filename_prefix, int(alpha * 100))
            _store_core_statistics_helper(filename, storages_alpha)


def _store_core_statistics_helper(filename, storages):
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(CoreStatistics.row_header())
        for storage in storages:
            writer.writerow(storage.as_row())


def load_core_statistics(name, filename_prefix="results"):
    if isinstance(name, str):
        filename = "{}-{}.csv".format(filename_prefix, name)
        storages = _load_core_statistics_helper(filename)
        return storages
    else:
        assert isinstance(name, list)
        storages_all = []
        for alpha in name:
            filename = "{}-at{}.csv".format(filename_prefix, int(alpha * 100))
            storages = _load_core_statistics_helper(filename)
            storages_all.append(storages)
        return storages_all


def _load_core_statistics_helper(filename):
    storages = []
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            cs = CoreStatistics.parse(row)
            storages.append(cs)
    return storages


def number_of_hidden_layers(model):
    return len(model.layers) - 2


def number_of_hidden_neurons(model):
    n = 0
    for layer_idx in range(1, len(model.layers) - 1):
        layer = model.layers[layer_idx]
        prod = 1
        for j in range(1, len(layer.output_shape)):
            prod *= layer.output_shape[j]
        n += prod
    return n
