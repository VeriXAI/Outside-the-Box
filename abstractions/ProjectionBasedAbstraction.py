from math import pow
from copy import deepcopy

from . import Abstraction, PartitionBasedAbstraction
from utils import *


class ProjectionBasedAbstraction(PartitionBasedAbstraction):
    def __init__(self, n: int, partition: list, abstractions, epsilon=0., epsilon_relative: bool = True, dim: int = -1):
        super().__init__(n, partition, abstractions, epsilon=epsilon, epsilon_relative=epsilon_relative, dim=dim)

    def name(self):
        return "ProjectionBasedAbstraction"

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
            return False, MAXIMUM_CONFIDENCE
        confidence = 0.0
        n_rejected = 0
        dim = 0
        for abstraction, block in zip(self.abstractions, self.partition):
            projected_vector = [vector[i] for i in range(dim, dim + block)]
            accepted, confidence_current = abstraction.isknown(projected_vector, skip_confidence=skip_confidence,
                                                               novelty_mode=novelty_mode)
            if not accepted:
                n_rejected += 1
                if skip_confidence:
                    break
                confidence += confidence_current
            dim += block
        if n_rejected > 0:
            accepts = False
            # the confidence is the average of all confidences (of rejecting blocks)
            confidence /= float(n_rejected)
        else:
            accepts = True
            confidence = ACCEPTANCE_CONFIDENCE
        return accepts, confidence

    def isempty(self):
        return self.abstractions[0].isempty()

    def closest_mean_dist(self, vector):
        distance_sos = 0.0
        dim = 0
        for abstraction, block in zip(self.abstractions, self.partition):
            projected_vector = [vector[i] for i in range(dim, dim + block)]
            distance_projection = abstraction.closest_mean_dist(projected_vector)
            distance_sos += pow(distance_projection, 2)  # undo the sqrt of Euclidean distance by squaring
            dim += block
        return sqrt(distance_sos)
