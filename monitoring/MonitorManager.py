from . import *
from utils import *


class MonitorManager(object):
    # --- public --- #

    def __init__(self, monitors: list, clustering_algorithm="KMeans", clustering_threshold=0.1, n_clusters=None,
                 filter_zeros=FILTER_ZERO_DIMENSIONS, alpha_thresholding=False, skip_confidence=True):
        self._monitors = monitors
        self._alpha_thresholding = alpha_thresholding
        self._layers = []
        self.clustering_algorithm = clustering_algorithm
        self.clustering_threshold = clustering_threshold
        self.n_clusters = n_clusters
        self.layer2class2nonzero_mask = dict() if filter_zeros else None
        self.skip_confidence = skip_confidence
        self.learn_from_test_data = any(m.is_test_training_active() for m in self._monitors)
        self.learn_from_novelty_data = any(m.is_novelty_training_active() for m in self._monitors)

    def layers(self):
        return self._layers

    def monitors(self):
        return self._monitors

    def normalize_and_initialize(self, model, n_classes):
        layers = set()
        for monitor in self._monitors:  # type: Monitor
            monitor.normalize_and_initialize(model, n_classes)
            layers.update(monitor.layers())
        self._layers = list(layers)
        print("Watching the following layers:")
        for layer in self._layers:
            print("- layer {:d} with {:d} neurons".format(layer, model.layers[layer].output_shape[1]))

    def train(self, model, data_train: DataSpec, data_test: DataSpec, statistics: Statistics,
              ignore_misclassifications=ONLY_LEARN_FROM_CORRECT_CLASSIFICATIONS):
        print("\n--- monitor training ---\n")

        # extract values for watched layers
        print("extracting data for watched layers")
        layer2values, timer = obtain_predictions(model=model, data=data_train, layers=self.layers(),
                                                 ignore_misclassifications=ignore_misclassifications)
        timer_sum = timer
        if self.learn_from_test_data or self.learn_from_novelty_data:
            layer2values_novel, timer = obtain_predictions(model=model, data=data_test, layers=self.layers())
            timer_sum += timer
            predictions_novel, timer = obtain_predictions(model=model, data=data_test)
            timer_sum += timer
        statistics.time_training_monitor_value_extraction = timer_sum

        # filter out zero dimensions
        if self.layer2class2nonzero_mask is not None:
            assert not self.learn_from_test_data
            self._determine_zero_filters(layer2values, model, data_train)
            layer2values = self._remove_zero_dimensions(layer2values, data_train.ground_truths(),
                                                        compute_violation_indices=False)
            for monitor in self._monitors:
                monitor.initialize_abstractions(self.layer2class2nonzero_mask)

        # clustering
        print("determining optimal clusters for each layer and class ({}, {})".format(
            self.clustering_threshold, self.n_clusters))
        layer2class2clusterer = self._clustering(data=data_train, layer2values=layer2values, statistics=statistics)
        if self.learn_from_test_data:
            data_train_combined = DataSpec(n=data_train.n + data_test.n, x=np.append(data_train.x(), data_test.x(), 0),
                                           y=np.append(data_train.y(), data_test.y(), 0))
            layer2values_combined = self._combine_layer2values(layer2values, layer2values_novel)
            layer2class2clusterer_combined = self._clustering(data=data_train_combined,
                                                              layer2values=layer2values_combined, statistics=statistics,
                                                              includes_test_data=True)

        # monitor training
        print("training monitors on the obtained data")
        self._train_monitors(data=data_train, layer2values=layer2values, layer2class2clusterer=layer2class2clusterer,
                             predictions=None, statistics=statistics, includes_test_data=False)
        if self.learn_from_test_data:
            self._train_monitors(data=data_train_combined, layer2values=layer2values_combined,
                                 layer2class2clusterer=layer2class2clusterer_combined, predictions=None,
                                 statistics=statistics, includes_test_data=True)

        if self.learn_from_novelty_data:
            # novelty training
            print("additional training of monitors on novelty data")
            self._train_monitors(data=data_test, layer2values=layer2values_novel, layer2class2clusterer=None,
                                 predictions=predictions_novel, statistics=statistics, includes_test_data=False)

    def run(self, model, data: DataSpec, statistics: Statistics):
        print("\n--- running monitored session ---\n")

        history = History()
        history.set_ground_truths(data.ground_truths())

        # extract values for watched layers and predictions of model
        layer2values, timer = obtain_predictions(model=model, data=data, layers=self.layers())
        timer_sum = timer
        predictions, timer = obtain_predictions(model=model, data=data)
        timer_sum += timer
        statistics.time_running_monitor_value_extraction = timer_sum
        history.set_layer2values(layer2values)
        history.set_predictions(predictions)

        # filter out zero dimensions
        if self.layer2class2nonzero_mask is not None:
            layer2values, zero_filter = self._remove_zero_dimensions(layer2values, predictions,
                                                                     compute_violation_indices=True)
            print("Found {:d} inputs that do not match the 'zero pattern'.".format(len(zero_filter)))
        else:
            zero_filter = []

        # run monitors
        timer = time()
        monitor2results = dict()
        for monitor in self._monitors:  # type: Monitor
            m_id = monitor.id()
            print("running monitor {:d} on the inputs".format(m_id))
            timer_monitor = time()
            monitor_results = monitor.run(layer2values=layer2values, predictions=predictions, history=history,
                                          zero_filter=zero_filter, skip_confidence=self.skip_confidence)
            statistics.time_running_each_monitor[m_id] = time() - timer_monitor
            monitor2results[m_id] = monitor_results
            if self._alpha_thresholding:
                self._compute_alpha_thresholding(monitor_results=monitor_results, model=model, data=data)
                monitor2results[-m_id] = monitor_results
                history.set_monitor_results(-m_id, monitor_results)
        statistics.time_running_monitor_classification = time() - timer

        return history

    # --- private --- #

    def _clustering(self, data, layer2values, statistics, includes_test_data=False):
        layers = self.layers()

        # cluster classes in each layer
        timer = time()
        layer2class2clusterer = dict()
        for layer in layers:
            class2values = dict()  # mapping: class_index -> values from watched layer
            values = layer2values[layer]
            for j, yj in enumerate(data.ground_truths()):
                vj = values[j]
                if yj in class2values:
                    class2values[yj].append(vj)
                else:
                    class2values[yj] = [vj]

            # find number of clusters
            print("Layer {:d}:".format(layer))
            class2clusters = cluster_refinement(class2values, algorithm=self.clustering_algorithm,
                                                threshold=self.clustering_threshold, n_clusters=self.n_clusters)
            layer2class2clusterer[layer] = class2clusters

            # update abstraction with number of clusters
            for monitor in self._monitors:  # type: Monitor
                if includes_test_data != monitor.is_test_training_active():
                    continue
                monitor.update_clustering(layer, class2clusters)

        statistics.time_training_monitor_clustering = time() - timer
        return layer2class2clusterer

    def _train_monitors(self, data, layer2values, layer2class2clusterer, predictions, statistics, includes_test_data):
        timer = time()
        ground_truths = data.ground_truths()
        novelty_training_mode = layer2class2clusterer is None
        for monitor in self._monitors:  # type: Monitor
            if (novelty_training_mode and not monitor.is_novelty_training_active()) or \
                    (includes_test_data != monitor.is_test_training_active()):
                continue
            print("training monitor {:d}{}".format(monitor.id(),
                                                   " with novelties" if layer2class2clusterer is None else ""))
            timer_monitor = time()
            if novelty_training_mode:
                monitor.train_with_novelties(predictions, layer2values)
            else:
                monitor.add_clustered(layer2values, ground_truths, layer2class2clusterer)
            duration = time() - timer_monitor
            if monitor.id() in statistics.time_tweaking_each_monitor.keys():
                statistics.time_tweaking_each_monitor[monitor.id()] += duration
            else:
                statistics.time_tweaking_each_monitor[monitor.id()] = duration
        duration = time() - timer
        if statistics.time_training_monitor_tweaking == -1:
            statistics.time_training_monitor_tweaking = duration
        else:
            statistics.time_training_monitor_tweaking += duration

    def _determine_zero_filters(self, layer2values: dict, model: Model, data: DataSpec):
        for layer, values in layer2values.items():
            n_neurons = model.layers[layer].output_shape[1]
            self.layer2class2nonzero_mask[layer] = determine_zero_filters(values, data, n_neurons, layer)

    def _remove_zero_dimensions(self, layer2values, classes, compute_violation_indices: bool):
        layer2values_new = dict()
        zero_indices = set()
        for layer, values in layer2values.items():
            class2nonzero_indices = self.layer2class2nonzero_mask[layer]
            filtered_values = []
            layer2values_new[layer] = filtered_values
            for j, (class_id, vj) in enumerate(zip(classes, values)):
                vj_filtered = []
                for vi, filter_i in zip(vj, class2nonzero_indices[class_id]):
                    if filter_i:
                        vj_filtered.append(vi)
                    elif vi != 0:
                        # found a violation
                        zero_indices.add(j)
                filtered_values.append(vj_filtered)
        if compute_violation_indices:
            return layer2values_new, sorted(zero_indices)
        else:
            return layer2values_new

    def _combine_layer2values(self, layer2values_1: dict, layer2values_2: dict):
        res = dict()
        for k, v in layer2values_1.items():
            res[k] = np.append(v, layer2values_2[k], axis=0)
        return res


    def _compute_alpha_thresholding(self, monitor_results: list, model, data: DataSpec):
        confidences_at = model.predict_proba(data.x())
        for monitor_result, prediction_model, confidence_at_vec in zip(monitor_results, confidences_at):
            monitor_result.add_confidence(np.max(confidence_at_vec))
