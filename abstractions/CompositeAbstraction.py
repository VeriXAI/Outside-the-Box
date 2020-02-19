from math import inf

from . import Abstraction
from utils import *


class CompositeAbstraction(Abstraction):
    def __init__(self, abstractions):
        if isinstance(abstractions[0], tuple):
            self.abstractions = [a(op) for a, op in abstractions]
        else:
            self.abstractions = abstractions

    def __str__(self):
        string = "["
        for i, abstraction in enumerate(self.abstractions):
            if i > 0:
                string += ", "
            string += str(abstraction)
        return string + "]"

    def long_str(self):
        string = "Composite: ["
        comma = ""
        for abstraction in self.abstractions:
            string += comma + abstraction.long_str()
            comma = ", "
        string += "]"
        return string

    def short_str(self):
        string = "Composite: ["
        comma = ""
        for abstraction in self.abstractions:
            string += comma + abstraction.short_str()
            comma = ", "
        string += "]"
        return string

    def __len__(self):
        return len(self.abstractions)

    def initialize(self, n_watched_neurons):
        for abstraction in self.abstractions:
            abstraction.initialize(n_watched_neurons)

    def add(self, class_id, vector):
        for abstraction in self.abstractions:
            abstraction.add(class_id, vector)

    def finalize(self):
        for abstraction in self.abstractions:
            abstraction.finalize()

    def isknown(self, vector, skip_confidence=False, novelty_mode=False):
        # average
        if COMPOSITE_ABSTRACTION_POLICY == 0:
            confidence_known = 0.0
            n_known = 0
            confidence_unknown = 0.0
            n_unknown = 0
            for abstraction in self.abstractions:
                result, confidence = abstraction.isknown(vector, skip_confidence=skip_confidence,
                                                         novelty_mode=novelty_mode)
                if result:
                    n_known += 1
                    confidence_known += confidence
                else:
                    n_unknown += 1
                    confidence_unknown += confidence

            # determine winner
            if n_known > 0:
                confidence_known /= float(n_known)
            if n_unknown > 0:
                confidence_unknown /= float(n_unknown)
            total_confidence = confidence_known + confidence_unknown  # normalization factor
            if confidence_known > confidence_unknown:
                return True, confidence_known / total_confidence
            else:
                return False, confidence_unknown / total_confidence
        # maximum
        elif COMPOSITE_ABSTRACTION_POLICY == 1:
            highest_confidence = -1.0
            for abstraction in self.abstractions:
                result, confidence = abstraction.isknown(vector)
                if result:
                    return result, confidence
                else:
                    highest_confidence = max(highest_confidence, confidence)
            return False, highest_confidence

        else:
            raise NotImplementedError("Policy {} is not available.".format(COMPOSITE_ABSTRACTION_POLICY))

    def clear(self):
        for abstraction in self.abstractions:
            abstraction.clear()

    def add_finalized(self, class_id, vector):
        for abstraction in self.abstractions:
            abstraction.add_finalized(class_id, vector)

    def default_options(self):
        return [(type(a), a.default_options()) for a in self.abstractions],  # needs to be a tuple

    def update_clustering(self, clusters):
        for abstraction in self.abstractions:
            abstraction.update_clustering(clusters)

    def add_clustered(self, values, clusterer):
        clusters = clusterer.predict(values)
        cluster2values = dict()
        for vj, cj in zip(values, clusters):
            if cj in cluster2values.keys():
                cluster2values[cj].append(vj)
            else:
                cluster2values[cj] = [vj]
        for cj, clustered_values in cluster2values.items():
            self.add_clustered_to_set(clustered_values, cj, self.mean_computer(clusterer, cj))

    def add_clustered_to_set(self, values, cj, mean_computer):
        for abstraction in self.abstractions:
            abstraction.add_clustered_to_set(self, values, cj, mean_computer)

    def closest_mean_dist(self, vector):
        min_distance = inf
        for abstraction in self.abstractions:
            min_distance = min(min_distance, abstraction.closest_mean_dist(vector))
        return min_distance

    def plot(self, dims, color, ax):
        for abstraction in self.abstractions:
            abstraction.plot(dims, color, ax)

    def isempty(self):
        for abstraction in self.abstractions:
            if not abstraction.isempty():
                return False
        return True
