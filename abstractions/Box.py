from copy import deepcopy
from matplotlib.patches import Rectangle
import random

from .PointCollection import PointCollection
from utils import *


class Box(PointCollection):
    def __init__(self, dimension):  # 'dimension' is ignored but should be present for interface reasons
        super().__init__()
        self.low = None
        self.high = None

    def __str__(self):
        if self.isempty():
            return "raw Box"
        return "  Box(l = " + str(self.low) + ",\n      h = " + str(self.high) + ")\n"

    def create(self, point):
        super().create(point)

        self.low = deepcopy(point)
        self.high = deepcopy(point)

    def contains(self, point, confidence_fun, bloating=0.0, bloating_relative=True, skip_confidence=False,
                 novelty_mode=False):
        inside = True
        if bloating == 0.0:
            for (i, pi) in enumerate(point):
                if not (self.low[i] <= pi <= self.high[i]):
                    inside = False
                    break
        else:
            assert bloating > 0, "bloating must be nonnegative"
            for (i, pi) in enumerate(point):
                bloating_distance = self.bloating_i(i, bloating, bloating_relative)
                if not (self.low[i] - bloating_distance <= pi <= self.high[i] + bloating_distance):
                    inside = False
                    break
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

        for (i, pi) in enumerate(point):
            if self.low[i] > pi:
                self.low[i] = pi
            elif pi > self.high[i]:
                self.high[i] = pi

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
            p1 = self.low[z]
            p2 = self.high[z]
            plot_interval(ax, p1, p2, color, epsilon, epsilon_relative, is_x_dim=y == -1)
            return

        left = self.low[x]
        bottom = self.low[y]
        width = self.high[x] - left
        height = self.high[y] - bottom

        if epsilon == 0 or PLOT_NON_EPSILON_SETS:
            rect = Rectangle((left, bottom), width, height, linewidth=1, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
        if epsilon == 0:
            return

        if epsilon_relative:
            # TODO relative increase of flat dimension has no effect
            x_eps = width / 2.0 * epsilon
            y_eps = height / 2.0 * epsilon
        else:
            x_eps = epsilon
            y_eps = epsilon
        left_eps = left - x_eps
        bottom_eps = bottom - y_eps
        width_eps = width + 2 * x_eps
        height_eps = height + 2 * y_eps
        line_style = "--" if PLOT_NON_EPSILON_SETS else "-"
        rect_eps = Rectangle((left_eps, bottom_eps), width_eps, height_eps, linewidth=1, edgecolor=color,
                             facecolor="none", linestyle=line_style)
        ax.add_patch(rect_eps)

    def diameter(self):
        return [(self.high[i] - li) for (i, li) in enumerate(self.low)]

    def center(self):
        return [((self.high[i] + li) / 2) for (i, li) in enumerate(self.low)]

    def dimension(self):
        return len(self.low)

    def bloating_i(self, i, epsilon, epsilon_relative):
        if epsilon_relative:
            return (self.high[i] - self.low[i]) / 2.0 * epsilon
        else:
            return epsilon

    def get_closest_point(self, point, epsilon, epsilon_relative):
        # the closest point can be determined component-wise
        closest_point = []
        for i, pi in enumerate(point):
            bloating = self.bloating_i(i, epsilon, epsilon_relative)
            if pi > self.high[i] + bloating:
                closest_point.append(self.high[i] + bloating)
            elif pi < self.low[i] - bloating:
                closest_point.append(self.low[i] - bloating)
            else:
                # point is inside the box in this dimension -> just use the point's position
                closest_point.append(pi)
        return closest_point

    def half_spaces(self, epsilon, epsilon_relative):
        return HalfSpaceIteratorBox(self, epsilon, epsilon_relative)


class HalfSpaceIteratorBox(object):
    def __init__(self, box: Box, epsilon, epsilon_relative):
        self.box = box
        self.epsilon = epsilon
        self.epsilon_relative = epsilon_relative
        self.i = 0
        self.low = True
        self.n = box.dimension()

    def __iter__(self):
        return self

    def __next__(self):
        i = self.i
        if i == self.n:
            raise StopIteration()
        a = [0.0 for _ in range(self.n)]
        bloating = self.box.bloating_i(i, self.epsilon, self.epsilon_relative)
        if self.low:
            a[i] = -1.0
            b = -self.box.low[i] + bloating
            self.low = False
        else:
            a[i] = 1.0
            b = self.box.high[i] + bloating
            self.low = True
            self.i += 1
        return a, b
