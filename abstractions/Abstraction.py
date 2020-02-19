from utils import cluster_center


class Abstraction(object):
    pass

    def short_str(self):
        return self.name()

    def name(self):
        return "Abstraction"

    def mean_computer(self, clusterer, cj):
        return lambda: cluster_center(clusterer, cj)
