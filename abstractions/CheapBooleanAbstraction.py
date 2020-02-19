import dd

from .Abstraction import Abstraction
from utils import *


# abstract a numeric vector to a bit-vector, which then is optionally abstracted again
#  example: {(5, -1, 2), (1, 1, 0)} ~> {(1, 0, 0), (1, 1, 0)}
# with the option 'single_vector == True' this intermediate abstraction is further abstracted to a bit-wise disjunction
#  example: {(1, 0, 0), (1, 1, 0)} ~> (1, 1, 0)
class CheapBooleanAbstraction(Abstraction):
    def __init__(self, dim=-1):
        self.bit_vector = None
        self.dim = dim

    def name(self):
        return "CheapBooleanAbstraction"

    def initialize(self, n_watched_neurons):
        self.dim = n_watched_neurons

    def add(self, vector):
        for i, (vi, pi) in enumerate(zip(vector, self.bit_vector)):
            if vi > 0 and pi == 0:
                self.bit_vector[i] = 1

    def isempty(self):
        return self.bit_vector is None

    def isknown(self, vector, skip_confidence=False, novelty_mode=False):
        if skip_confidence:
            if novelty_mode:
                confidence = SKIPPED_CONFIDENCE_NOVELTY_MODE
            else:
                confidence = SKIPPED_CONFIDENCE
        if self.isempty():
            return False
        accepts = True
        for vi, pi in zip(vector, self.bit_vector):
            if vi > 0 and pi == 0:
                accepts = False
                break
        if accepts:
            confidence = ACCEPTANCE_CONFIDENCE
        else:
            confidence = MAXIMUM_CONFIDENCE
        return accepts, confidence

    def add_clustered(self, values, clusterer=None):
        self.bit_vector = [0 for _ in range(self.dim)]
        for vj in values:
            self.add(vj)

    def update_clustering(self, clusters):
        pass

    def plot(self, dims, color, ax):
        pass
