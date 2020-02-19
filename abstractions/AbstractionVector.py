from copy import deepcopy


class AbstractionVector(object):
    def __init__(self, abstraction, n_classes):
        self._abstractions = [deepcopy(abstraction) for i in range(n_classes)]

    def __str__(self):
        return str(self._abstractions[0])

    def abstractions(self):
        return self._abstractions

    def long_str(self):
        string = str(self)
        for i, abstraction in enumerate(self._abstractions):
            string += "\n class", i, "-> " + abstraction.long_str()
        return string

    def short_str(self):
        return self._abstractions[0].short_str()

    def initialize(self, n_watched_neurons):
        for abstraction in self._abstractions:
            abstraction.initialize(n_watched_neurons)

    def add(self, class_id, vector):
        self._abstractions[class_id].add(vector)

    def finalize(self):
        for abstraction in self._abstractions:
            if not abstraction.isempty():
                abstraction.finalize()

    def isknown(self, class_id, vector, skip_confidence=False, novelty_mode=False):
        return self._abstractions[class_id].isknown(vector, skip_confidence=skip_confidence or novelty_mode,
                                                    novelty_mode=novelty_mode)

    def clear(self):
        for abstraction in self._abstractions:
            abstraction.clear()

    def add_finalized(self, class_id, vector):
        self._abstractions[class_id].add_finalized(vector)

    def default_options(self):
        return self._abstractions[0].default_options()

    def coarsen_options(self, options):
        return self._abstractions[0].coarsen_options(options)

    def refine_options(self, options):
        return self._abstractions[0].refine_options(options)

    def propose(self, vector):
        # proposal is only based on mean
        class_proposed = -1
        min_distance = float("inf")
        for class_index, abstraction in enumerate(self._abstractions):
            if abstraction.isempty():
                continue
            distance = abstraction.closest_mean_dist(vector)
            if distance < min_distance:
                min_distance = distance
                class_proposed = class_index
        assert class_proposed >= 0, "Did not find any nonempty abstraction."
        return class_proposed

    def update_clustering(self, class_index, clusters):
        self._abstractions[class_index].update_clustering(clusters)

    def add_clustered(self, class_index, values, clusters):
        self._abstractions[class_index].add_clustered(values, clusters)
