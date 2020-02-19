from .SetBasedAbstraction import SetBasedAbstraction
from .ConvexHull import ConvexHull
from utils import *


class ConvexHullAbstraction(SetBasedAbstraction):
    def __init__(self, confidence_fun, size=1, epsilon=0., epsilon_relative=False,
                 remove_redundancies=CONVEX_HULL_REDUNDANCY_REMOVAL):
        super().__init__(confidence_fun, size, epsilon, epsilon_relative)
        self.remove_redundancies = remove_redundancies

    def name(self):
        return "ConvexHull"

    def set_type(self):
        return ConvexHull

    def add_clustered_to_set(self, values, cj, mean_computer):
        _set = self.sets[cj]  # type: ConvexHull
        assert _set.isempty()
        _set.create(values, remove_redundancies=self.remove_redundancies)
