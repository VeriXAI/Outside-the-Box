from scipy.spatial.distance import euclidean

from utils import *


class PointCollection(object):
    def __init__(self):
        self.sum = None
        self.n_points = 0
        self._mean = None
        self.n_novelty_points = 0
        self._incredibility = None

    def create(self, point):
        if COMPUTE_MEAN:
            self.sum = point
        self.n_points = 1

    def isempty(self):
        return self.n_points == 0

    def add_novelty_point(self):
        self.n_novelty_points += 1

    def compute_credibility(self, n_total):
        if not self.isempty():
            num = self.n_novelty_points
            den = self.n_points + self.n_novelty_points
            # den = n_total
            self._incredibility = float(num) / float(den)
            if PRINT_CREDIBILITY:
                print("incredibility: {:d}/{:d} = {} %".format(num, den, self._incredibility))

    def add(self, point):
        if COMPUTE_MEAN:
            for i, pi in enumerate(point):
                self.sum[i] += pi
        self.n_points += 1

    def mean(self):
        if self._mean is None:
            assert COMPUTE_MEAN, "Mean computation was deactivated!"
            self._mean = [pi / self.n_points for pi in self.sum]
        return self._mean

    def center(self):
        return self.mean()

    def euclidean_distance(self, point, epsilon, epsilon_relative):
        closest_point = self.get_closest_point(point, epsilon, epsilon_relative)
        assert list(point) != closest_point, "Confidence for points inside the set should not be asked for!"
        dist = euclidean(point, closest_point)
        radius = euclidean(closest_point, self.center())
        if radius == 0.0:
            # corner case: the set consists of a single point only
            confidence = dist
        else:
            # normalization so that confidence 1.0 corresponds to dist == radius
            confidence = dist / radius
        return confidence

    def get_closest_point(self, point, epsilon, epsilon_relative):
        raise NotImplementedError("get_closest_point() is not implemented by {}".format(type(self)))

    def halfspace_distance(self, point, epsilon, epsilon_relative):
        if self._is_corner_case():
            return CONVEX_HULL_HALF_SPACE_DISTANCE_CORNER_CASE
        highest_distance = -1.0
        for a, b in self.half_spaces(epsilon, epsilon_relative):
            distance = b - np.dot(a, point)
            highest_distance = max(highest_distance, distance)
        assert highest_distance > 0.0, "Confidence for points inside the set should not be asked for!"
        return highest_distance

    def half_spaces(self, epsilon, epsilon_relative):
        raise NotImplementedError("half_spaces() is not implemented by {}".format(type(self)))

    def _is_corner_case(self):
        # NOTE: only to be overridden by ConvexHull
        return False
