import dd
from scipy.spatial.distance import euclidean as dist_fun

from .Abstraction import Abstraction
from .PointCollection import PointCollection
from utils import *


def _var(i):
    return 'x' + str(i)


class BooleanAbstraction(Abstraction):
    def __init__(self, gamma=0):
        self.bdd = dd.BDD()
        self.formula = None
        self.point_collection = PointCollection()
        self.dim = -1
        self.gamma = gamma

    def __str__(self):
        return "Bit-vector (γ = " + str(self.gamma) + ")"

    def name(self):
        return "BooleanAbstraction"

    def long_str(self):
        string = str(self) + " with " + str(self.bdd.to_expr(self.formula)) + "\n"
        return string

    def initialize(self, n_watched_neurons):
        self.dim = n_watched_neurons
        for i in range(n_watched_neurons):
            self.bdd.declare(_var(i))

    def create(self, vector):
        self.add(vector, create=True)

    def add(self, vector, create=False):
        if create:
            self.point_collection.create(vector)
        else:
            self.point_collection.add(vector)

        a = self._to_bit_vector(vector)
        conjunction = self._to_conjunction(a)
        f = self.bdd.add_expr(conjunction)
        g = self.formula
        if g is None:
            g = f
            self.point_collection.create(vector)
        else:
            g = g | f
            self.point_collection.add(vector)
        for _ in range(self.gamma):
            g_prev = g
            for i in range(self.dim):
                g = g | self.bdd.exist([_var(i)], g_prev)
        self.formula = g

    def isempty(self):
        return self.point_collection.isempty()

    def isknown(self, vector, skip_confidence=False, novelty_mode=False):
        if skip_confidence:
            if novelty_mode:
                confidence = SKIPPED_CONFIDENCE_NOVELTY_MODE
            else:
                confidence = SKIPPED_CONFIDENCE
        if self.isempty():
            return False
        a = self._to_bit_vector(vector)
        conjunction = self._to_conjunction(a)
        f = self.bdd.add_expr(conjunction)
        g = self.formula & f
        accepts = self.bdd.pick(g) is not None
        if accepts:
            confidence = ACCEPTANCE_CONFIDENCE
        else:
            confidence = MAXIMUM_CONFIDENCE
        return accepts, confidence

    @staticmethod
    def _to_bit_vector(vector):
        a = [0] * len(vector)
        for i, v in enumerate(vector):
            if v <= 0:
                a[i] = 0
            else:
                a[i] = 1
        return a

    @staticmethod
    def _to_conjunction(a):
        conjunction = ""
        for i, v in enumerate(a):
            if i > 0:
                conjunction += " & "
            if v == 0:
                conjunction += "~x"
            else:
                conjunction += "x"
            conjunction += str(i)
        return conjunction

    def clear(self):
        self.bdd = dd.BDD()
        self.formula = None
        assert self.dim > 0, "Boolean abstraction has not been initialized yet."
        self.initialize(self.dim)

    def add_finalized(self, vector):
        self.add(vector)

    @staticmethod
    def default_options():
        gamma = 0
        return gamma,  # needs to be a tuple

    def closest_mean_dist(self, vector):
        # there is only one mean
        _mean = self.point_collection.mean()
        return dist_fun(_mean, vector)

    def update_clustering(self, clusters):
        pass  # no clustering for Boolean abstraction

    def add_clustered(self, values, clusterer):
        # no clustering for Boolean abstraction
        self.add_clustered_to_set(values, 0, self.mean_computer(clusterer, 0))

    def add_clustered_to_set(self, values, cj, mean_computer):
        assert cj == 0
        # create BDD for γ = 0
        g = self.formula
        for vj in values:
            a = self._to_bit_vector(vj)
            conjunction = self._to_conjunction(a)
            f = self.bdd.add_expr(conjunction)
            if g is None:
                g = f
                self.point_collection.create(vj)
            else:
                g = g | f
                self.point_collection.add(vj)

        # modify BDD for γ > 0
        dim = len(values[0])
        g_prev = g
        for i in range(self.gamma):
            # initialize g to empty set
            g = self.bdd.false

            # compute extension for index i based on index i-1
            for j in range(self.dim):
                g = g | self.bdd.exist([_var(j)], g_prev)
            g_prev = g

        # update the BDD with the new formula
        self.formula = g

    def plot(self, dims, color, ax):
        return  # cannot plot Boolean abstraction
