from math import pow
from copy import deepcopy

from . import Abstraction
from utils import *


class PartitionBasedAbstraction(Abstraction):
    def __init__(self, n: int, partition: list, abstractions, epsilon=0., epsilon_relative: bool = True, dim: int = -1,
                 fixed_n: bool = False):
        self.dim = dim
        self.epsilon = epsilon
        self.epsilon_relative = epsilon_relative
        self.n_sets = n
        self.partition = partition  # a block is represented by its size
        if isinstance(abstractions, Abstraction):  # convenience functionality: use the same abstraction for each block
            abstractions = [deepcopy(abstractions) for _ in range(len(partition))]
        self.abstractions = abstractions
        self.fixed_n = fixed_n

    def name(self):
        return "PartitionBasedAbstraction"

    def short_str(self):
        string = "Partition: ["
        comma = ""
        for abstraction, block in zip(self.abstractions, self.partition):
            string += comma + str(block) + "D " + abstraction.short_str()
            comma = ", "
        string += "]"
        return string

    def initialize(self, n_watched_neurons, n_sets=None):
        # each abstraction uses the same number of sets
        if n_sets is None:
            n_sets = self.n_sets
        self._adapt_partition(n_watched_neurons)
        for abstraction, block in zip(self.abstractions, self.partition):
            abstraction.initialize(block, n_sets=n_sets)
        self.dim = n_watched_neurons

    def update_clustering(self, clusters):
        # each abstraction uses the same clusters
        n_sets = None if self.fixed_n else cluster_number(clusters)
        if n_sets != self.n_sets:
            self.initialize(n_watched_neurons=self.dim, n_sets=n_sets)

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
        dim = 0
        for abstraction, block in zip(self.abstractions, self.partition):
            projected_values = [np.array([vector[i] for i in range(dim, dim + block)]) for vector in values]
            projected_mean_computer = self.projected_mean_computer(mean_computer, dim, dim + block)
            abstraction.add_clustered_to_set(projected_values, cj, projected_mean_computer)
            dim += block

    def isknown(self, vector, skip_confidence=False, novelty_mode=False):
        if self.isempty():
            return True, MAXIMUM_CONFIDENCE
        lowest_confidence = math.inf
        for j in range(len(self.abstractions[0].sets)):
            # try to match set j
            highest_confidence = -1.0
            dim = 0
            for abstraction, block_size in zip(self.abstractions, self.partition):
                projected_vector = [vector[i] for i in range(dim, dim + block_size)]
                accepted, confidence_current = abstraction.isknown_given_set_index(projected_vector, j,
                                                                                   skip_confidence=skip_confidence,
                                                                                   novelty_mode=novelty_mode)
                if not accepted:
                    highest_confidence = max(highest_confidence, confidence_current)
                    if highest_confidence >= MAXIMUM_CONFIDENCE or skip_confidence:
                        break
                dim = dim + block_size
            if highest_confidence == -1.0:
                return True, ACCEPTANCE_CONFIDENCE
            lowest_confidence = min(lowest_confidence, highest_confidence)
        return False, lowest_confidence

    def isempty(self):
        return self.abstractions[0].isempty()

    def closest_mean_dist(self, vector):
        smallest_distance = 0.0
        for j in range(len(self.abstractions[0].sets)):
            # try to match set j
            distance_sos = 0.0
            dim = 0
            for abstraction, block in zip(self.abstractions, self.partition):
                projected_vector = [vector[i] for i in range(dim, dim + block)]
                distance_projection = abstraction.closest_mean_dist_given_set_index(projected_vector, j)
                distance_sos += pow(distance_projection, 2)  # undo the sqrt of Euclidean distance by squaring
                dim += block
            smallest_distance = min(distance_sos, smallest_distance)
        return sqrt(smallest_distance)

    def projected_mean_computer(self, mean_computer, start, end):
        mean = mean_computer()
        return lambda: [mean[i] for i in range(start, end)]

    def plot(self, dims, color, ax):
        x = dims[0]
        y = dims[1]
        if x == -1 and y == -1:
            plot_zero_point(ax, color, epsilon=0, epsilon_relative=False)
            return
        elif x == -1 or y == -1:
            if x == -1:
                z = y
            else:
                z = x
        else:
            z = x
        # find relevant block
        block_index = 0
        dim = 0
        for block in self.partition:
            assert dim <= z
            if z < dim + block:
                if x != -1 and y != -1:
                    assert dim <= y < dim + block, "Can only plot dimensions from the same block in a partition."
                break
            dim += block
            block_index += 1

        abstraction = self.abstractions[block_index]
        if x >= 0:
            x -= dim
        if y >= 0:
            y -= dim
        abstraction.plot([x, y], color, ax)

    def _adapt_partition(self, n):
        i = 0
        for ib, block in enumerate(self.partition):
            i += block
            if i >= n:
                del self.partition[ib + 1:]
                if i > n:
                    self.partition[ib] -= i - n
                break
