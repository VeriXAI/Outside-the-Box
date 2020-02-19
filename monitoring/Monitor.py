import itertools
from copy import deepcopy

from . import *
from utils import *
from abstractions import *


class Monitor(object):
    """
    A monitor consists of layer-abstraction mappings. It can evaluate a given input based on its abstractions.

    The acceptance behavior of a monitor is defined in the class MonitorResult.

    Fields:
     - _layer2abstraction: mapping 'layer → abstraction'
     - _score_fun: score function for training (see Score class)
                   The value 'None' means that this monitor is not trained.
                   default: AverageScore()
    """

    _id_iter = itertools.count()

    # --- public --- #

    def __init__(self, layer2abstraction: dict, score_fun=AverageScore(), layer2dimensions=None,
                 learn_from_test_data=False, is_novelty_training_active=False):
        self._id = next(Monitor._id_iter)
        if self._id == 0:
            self._id = next(Monitor._id_iter)  # start with '1'
        self._layer2abstraction = layer2abstraction
        self._score_fun = score_fun
        if layer2dimensions is None:
            layer2dimensions = {layer: [0, 1] for layer in self._layer2abstraction.keys()}
        self._layer2dimensions = layer2dimensions
        self._layer2class2dimensions = None
        self._learn_from_test_data = learn_from_test_data
        self._is_novelty_training_active = is_novelty_training_active

    @staticmethod
    def reset_ids():
        Monitor._id_iter = itertools.count()

    def __str__(self):
        return "Monitor {:d}".format(self.id())

    def id(self):
        return self._id

    def layers(self):
        return self._layer2abstraction.keys()

    def abstraction(self, layer):
        return self._layer2abstraction[layer]

    def short_str(self):
        string = ""
        for l, a in self._layer2abstraction.items():
            if string != "":
                string += ", "
            string += "layer {:d}: {}".format(l, a.short_str())
        return string

    def long_str(self):
        string = ""
        for l, a in self._layer2abstraction.items():
            if string != "":
                string += ", "
            string += "layer {:d}: {}".format(l, a.long_str())
        return string

    def dimensions(self, layer, class_id=None):
        if class_id is not None and self._layer2class2dimensions is not None:
            return self._layer2class2dimensions[layer][class_id]
        return self._layer2dimensions[layer]

    def normalize_and_initialize(self, model, n_classes):
        layer2abstraction_new = dict()
        for layer, abstraction in self._layer2abstraction.items():  # type: int, Abstraction
            # normalize layer index
            layer_normalized = normalize_layer(model, layer)

            # obtain number of neurons
            n_neurons = model.layers[layer_normalized].output_shape[1]

            # normalize abstraction (wrap in AbstractionVectors)
            if isinstance(abstraction, AbstractionVector):
                assert len(abstraction._abstractions) == n_classes, "Detected wrong number of abstractions!"
                abstraction_new = abstraction
            else:
                abstraction_new = AbstractionVector(abstraction, n_classes)

            # initialize abstraction
            abstraction_new.initialize(n_neurons)

            # update new mapping
            if layer_normalized in layer2abstraction_new:
                raise(ValueError("Duplicate layer index", layer_normalized, "found. Please use unique indexing."))
            layer2abstraction_new[layer_normalized] = abstraction_new
        self._layer2abstraction = layer2abstraction_new

        layer2dimensions_new = dict()
        for layer, dimensions in self._layer2dimensions.items():  # type: int, list
            # normalize layer index
            layer_normalized = normalize_layer(model, layer)
            if layer_normalized in layer2dimensions_new:
                raise(ValueError("Duplicate layer index", layer_normalized, "found. Please use unique indexing."))
            layer2dimensions_new[layer_normalized] = dimensions
        self._layer2dimensions = layer2dimensions_new

    def initialize_abstractions(self, layer2class2nonzero_mask):
        self._layer2class2dimensions = dict()
        for layer, abstraction_vector in self._layer2abstraction.items():  # type: int, AbstractionVector
            class2dimensions = dict()
            self._layer2class2dimensions[layer] = class2dimensions
            original_dimensions = self._layer2dimensions[layer]
            class2nonzero_mask = layer2class2nonzero_mask[layer]  # type: dict
            for class_id, nonzero_mask in class2nonzero_mask.items():
                abstraction = abstraction_vector._abstractions[class_id]
                abstraction.initialize(sum([1 if nonzero else 0 for nonzero in nonzero_mask]))
                # adapt plotting dimension
                dimensions = []
                for dim in [0, 1]:
                    res = original_dimensions[dim]
                    if not nonzero_mask[res]:
                        res = -1
                    else:
                        res -= sum(not is_nz for is_nz in nonzero_mask[:res + 1])
                    dimensions.append(res)
                class2dimensions[class_id] = dimensions

    def update_clustering(self, layer: int, class2clusters: dict):
        abstraction_vector = self._layer2abstraction.get(layer)
        if abstraction_vector is None:
            # this monitor does not watch the given layer
            return

        assert isinstance(abstraction_vector, AbstractionVector)
        for class_index, clusters in class2clusters.items():
            abstraction_vector.update_clustering(class_index, clusters)

    def add_clustered(self, layer2values, ground_truths, layer2class2clusterer):
        for layer, abstraction_vector in self._layer2abstraction.items():
            values = layer2values[layer]

            # mapping: class_index -> values from watched layer
            class2values = dict()
            for j, yj in enumerate(ground_truths):
                vj = values[j]
                if yj in class2values:
                    class2values[yj].append(vj)
                else:
                    class2values[yj] = [vj]

            class2clusters = layer2class2clusterer[layer]
            for class_index, values in class2values.items():
                clusterer = class2clusters[class_index]
                values_copy = deepcopy(values)  # for some reason, the list is modified below
                abstraction_vector.add_clustered(class_index, values_copy, clusterer)

    def train_with_novelties(self, predictions: list, layer2values: dict):
        for layer, abstraction in self._layer2abstraction.items():
            for pj, vj in zip(predictions, layer2values[layer]):
                abstraction.isknown(pj, vj, novelty_mode=True)
        for abstraction_vector in self._layer2abstraction.values():  # type: AbstractionVector
            for abstraction in abstraction_vector.abstractions():
                abstraction.compute_credibility(len(predictions))

    def run(self, layer2values: dict, predictions: list, history: History, zero_filter: list, skip_confidence=False):
        results = [MonitorResult() for _ in predictions]
        for layer, abstraction in self._layer2abstraction.items():
            if zero_filter:
                zero_filter_index = 0
                zero_filter_value = zero_filter[0]
            else:
                zero_filter_index = -1
                zero_filter_value = -1
            for j, vj in enumerate(layer2values[layer]):
                if j == zero_filter_value:
                    results[j].set_zero_filter()

                    # find next zero index
                    zero_filter_index += 1
                    if zero_filter_index == len(zero_filter):
                        zero_filter_value = -1
                    else:
                        zero_filter_value = zero_filter[zero_filter_index]
                else:
                    c_predicted = predictions[j]
                    accepts, confidence = abstraction.isknown(c_predicted, vj, skip_confidence=skip_confidence)
                    results[j].add_confidence(confidence)
        history.set_monitor_results(m_id=self.id(), results=results)
        return results

    def is_novelty_training_active(self):
        return self._is_novelty_training_active

    def is_test_training_active(self):
        return self._learn_from_test_data
