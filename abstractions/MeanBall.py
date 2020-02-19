from matplotlib.patches import Circle
from scipy.spatial.distance import euclidean

from .PointCollection import PointCollection
from utils import *


class MeanBall(PointCollection):
    def __init__(self, dimension):
        super().__init__()
        self.center = [0.0 for _ in range(dimension)]
        self.radius = 0.0

    def __str__(self):
        if self.isempty():
            return "raw MeanBall"
        return "  MeanBall(r={:.4f})".format(self.radius)

    def create(self, point, center=None):
        super().create(point)
        if center is None:
            self.center = point
            self.radius = 0.0
        else:
            self.center = center
            self.radius = euclidean(center, point)

    def contains(self, point, confidence_fun, bloating=0.0, bloating_relative=True, skip_confidence=False,
                 novelty_mode=False):
        assert bloating >= 0, "bloating must be nonnegative"
        radius = self._radius(bloating, bloating_relative)
        inside = euclidean(self.center, point) <= radius
        if inside:
            confidence = ACCEPTANCE_CONFIDENCE
            if novelty_mode:
                self.add_novelty_point()
            elif self._incredibility is not None and random.random() < self._incredibility:
                inside = False
                confidence = INCREDIBLE_CONFIDENCE
        elif skip_confidence:
            if novelty_mode:
                confidence = SKIPPED_CONFIDENCE_NOVELTY_MODE
            else:
                confidence = SKIPPED_CONFIDENCE
        else:
            confidence = confidence_fun(self, point, bloating, bloating_relative)
        return inside, confidence

    def add(self, point):
        super().add(point)
        distance = euclidean(self.center, point)
        if distance > self.radius:
            self.radius = distance

    def plot(self, dims, color, epsilon, epsilon_relative, ax):
        x = dims[0]
        y = dims[1]
        if x == -1 and y == -1:
            plot_zero_point(ax, color, epsilon, epsilon_relative)
            return
        elif x == -1 or y == -1:
            if x == -1:
                z = y
            else:
                z = x
            p1 = self.center[x] - self.radius
            p2 = self.center[x] + self.radius
            plot_interval(ax, p1, p2, color, epsilon, epsilon_relative, is_x_dim=y == -1)
            return

        center = (self.center[x], self.center[y])
        if epsilon == 0 or PLOT_NON_EPSILON_SETS:
            circle = Circle(center, radius=self.radius, linewidth=1, edgecolor=color, facecolor="none")
            ax.add_patch(circle)
        if epsilon == 0:
            return

        radius = self._radius(epsilon, epsilon_relative)
        line_style = "--" if PLOT_NON_EPSILON_SETS else "-"
        circle = Circle(center, radius=radius, linewidth=1, edgecolor=color, facecolor="none", linestyle=line_style)
        ax.add_patch(circle)

    def dimension(self):
        return len(self.center)

    def center(self):
        return self.center

    def mean(self):
        return self.center

    def _radius(self, bloating, bloating_relative):
        if bloating_relative:
            return self.radius * (1 + bloating)
        else:
            return self.radius + bloating

    # --- distances ---

    def euclidean_distance(self, point, epsilon, epsilon_relative):
        dist = euclidean(self.center, point)
        radius = self._radius(epsilon, epsilon_relative)
        assert dist >= radius, "Confidence for points inside the set should not be asked for!"
        if radius == 0.0:
            # corner case: the ball consists of a single point only
            confidence = dist
        else:
            # normalization so that confidence 1.0 corresponds to dist == 2 * radius
            confidence = (dist - radius) / radius
        return confidence
