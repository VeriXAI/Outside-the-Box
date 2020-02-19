from scipy.spatial.distance import euclidean  # NOTE: changing this requires adapting PartitionBasedAbstraction as well
from math import inf

from .Abstraction import Abstraction
from . import PointCollection
from utils import *

EPSILON_STEP = 0.5


class SetBasedAbstraction(Abstraction):
    def __init__(self, confidence_fun, n, epsilon=0., epsilon_relative=True, dim=-1):
        self.sets = [None for _ in range(n)]
        self.dim = dim
        self.epsilon = epsilon
        self.epsilon_relative = epsilon_relative
        self.confidence_fun = confidence_fun

    def __str__(self):
        return self.long_str()

    def name(self):
        return "SetBasedAbstraction"

    def set_type(self):
        raise(NotImplementedError("This method must be implemented!"))

    def long_str(self):
        return self.name() + " (" + str(len(self.sets)) + " sets, ε_" +\
               ("rel" if self.epsilon_relative else "abs") + " = " + str(self.epsilon) + ")"

    def initialize(self, n_watched_neurons, n_sets=None):
        if n_sets is None:
            n_sets = len(self.sets)
        self.sets[:] = [self.set_type()(n_watched_neurons) for _ in range(n_sets)]
        self.dim = n_watched_neurons

    def add(self, vector):
        for _set in self.sets:
            if _set.isempty():
                # TODO this is naive
                print("Warning: using a naive 'add' method for the abstraction - better use 'add_clustered'.")
                # found a free set; all further sets are also free
                _set.create(vector)
                return
            elif _set.contains(vector, bloating=self.epsilon):
                # vector is already contained
                return
        # choose a set for adding the vector to
        _set = self.sets[self._choose_set_index(vector)]
        # TODO allow to merge sets?
        # if _set.isempty():
        #     _set.create(vector)
        # else:
        _set.add(vector)

    def isempty(self):
        return not self.sets or self.sets[0].isempty()

    def isknown_given_set_index(self, vector, set_index, skip_confidence=False, novelty_mode=False):
        set = self.sets[set_index]
        if not set.isempty():
            result, confidence = set.contains(vector, self.confidence_fun, bloating=self.epsilon,
                                              bloating_relative=self.epsilon_relative, skip_confidence=skip_confidence,
                                              novelty_mode=novelty_mode)
            if result:
                return True, ACCEPTANCE_CONFIDENCE
            else:
                return False, confidence
        else:
            return False, MAXIMUM_CONFIDENCE

    def isknown(self, vector, skip_confidence=False, novelty_mode=False):
        best_result = False
        if not self.isempty():
            lowest_confidence = math.inf
            for set_index in range(len(self.sets)):
                result, confidence = self.isknown_given_set_index(vector, set_index, skip_confidence=skip_confidence,
                                                                  novelty_mode=novelty_mode)
                if result and not novelty_mode:
                    lowest_confidence = ACCEPTANCE_CONFIDENCE
                    best_result = True
                    break
                lowest_confidence = min(lowest_confidence, confidence)
        else:
            lowest_confidence = MAXIMUM_CONFIDENCE
        return best_result, lowest_confidence

    def _choose_set_index(self, vector):
        min_distance = float('Inf')
        min_distance_index = -1
        for i, set_i in enumerate(self.sets):
            if set_i.isempty():
                continue
            dist = euclidean(set_i.mean(), vector)
            if dist < min_distance:
                min_distance_index = i
                min_distance = dist
        return min_distance_index

    def add_finalized(self, vector):
        self.add(vector)

    @staticmethod
    def default_options():
        n = 1
        epsilon = 0.
        return n, epsilon

    @staticmethod
    def coarsen_options(options):
        n, epsilon = options
        epsilon += EPSILON_STEP
        return n, epsilon

    @staticmethod
    def refine_options(options):
        n, epsilon = options
        n += 1
        epsilon = max(0, epsilon - EPSILON_STEP)
        return n, epsilon

    def clear(self):
        self.sets = [self.set_type()(self.dim) for _ in self.sets]

    def plot(self, dims, color, ax):
        for _set in self.sets:
            if _set.isempty():
                continue
            # plot set
            _set.plot(dims, color, self.epsilon, self.epsilon_relative, ax)
            # plot mean
            if PLOT_MEAN and COMPUTE_MEAN:
                _mean = _set.mean()
                ax.plot(_mean[dims[0]], _mean[dims[1]], c=color, marker="+")

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
        if len(values) == 0:
            return
        _set = self.sets[cj]
        assert _set.isempty()
        _set.create(values[0])
        for vj in values[1:]:
            _set.add(vj)

    def update_clustering(self, clusters):
        n_sets = cluster_number(clusters)
        if n_sets != len(self.sets):
            self.initialize(n_watched_neurons=self.dim, n_sets=n_sets)

    def get_sets(self):
        return self.sets

    def closest_mean_dist_given_set_index(self, vector, set_index):
        _set = self.sets[set_index]
        if _set.isempty():
            return inf
        _mean = _set.mean()
        return euclidean(_mean, vector)

    def closest_mean_dist(self, vector):
        min_distance = inf
        for set_index in range(len(self.sets)):
            distance = self.closest_mean_dist_given_set_index(vector, set_index)
            min_distance = min(min_distance, distance)
        return min_distance

    def compute_credibility(self, n_total):
        for _set in self.sets:
            _set.compute_credibility(n_total)
