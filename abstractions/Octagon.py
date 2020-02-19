from matplotlib.patches import Polygon
from itertools import chain

from .PointCollection import PointCollection
from .Box import *
from utils import *


"""
Octagons represent a higher-dimensional generalization of regular octagons. They are similar to zones, just with the
"missing" facets.
An octagon is represented by two DBMs (which are stored in a row-major way) and two boxes.
An entry 'M[i][j] = v' represents the constraint 'x_i - x_j <= v'.
"""
class Octagon(PointCollection):
    def __init__(self, dimension):
        super().__init__()
        self.dbm_sum = [[] for _ in range(dimension)]
        self.dbm_difference = [[] for _ in range(dimension)]
        self.box = Box(dimension)

    def __str__(self):
        if self.isempty():
            return "raw Octagon"
        return "  Octagon\n"

    def create(self, point):
        super().create(point)

        # create box constraints
        self.box.create(point)

        # create diagonal constraints
        for i, pi in enumerate(point):
            row_sum = self.dbm_sum[i]
            row_difference = self.dbm_difference[i]
            for j, pj in enumerate(point):
                sum = pi + pj
                if i <= j:
                    row_sum.append(sum)  # xi + xj <= c
                else:
                    row_sum.append(-sum)  # xi + xj >= c  (== -(xi + xj) <= -c)
                row_difference.append(pi - pj)  # xi - xj <= c

    def contains(self, point, confidence_fun, bloating=0.0, bloating_relative=True, skip_confidence=False,
                 novelty_mode=False):
        assert bloating >= 0, "bloating must be nonnegative"
        # check box constraints
        inside, confidence = self.box.contains(point, confidence_fun, bloating, bloating_relative, skip_confidence=True,
                                               novelty_mode=novelty_mode)

        if inside:
            # check diagonal constraints
            n = len(point)
            if bloating_relative:
                for i, pi in enumerate(point):
                    for j in range(i + 1, n):
                        pj = point[j]
                        bloating_distance = abs(self.dbm_difference[i][j] - self.dbm_difference[j][i]) / 2.0 * bloating
                        if not (-self.dbm_difference[j][i] - bloating_distance <= pi - pj <=
                                self.dbm_difference[i][j] + bloating_distance):
                            inside = False
                            break
                        bloating_distance = abs(self.dbm_sum[i][j] - self.dbm_sum[j][i]) / 2.0 * bloating
                        if not (-self.dbm_sum[j][i] - bloating_distance <= pi + pj <=
                                self.dbm_sum[i][j] + bloating_distance):
                            inside = False
                            break
                    if not inside:
                        break
            else:
                for i, pi in enumerate(point):
                    for j in range(i + 1, n):
                        pj = point[j]
                        if not (self.dbm_difference[j][i] - bloating <= pi - pj <=
                                self.dbm_difference[i][j] + bloating):
                            inside = False
                            break
                        if not (self.dbm_sum[j][i] - bloating <= pi + pj <= self.dbm_sum[i][j] + bloating):
                            inside = False
                            break
                    if not inside:
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

        # add to box constraints
        self.box.add(point)

        # add to diagonal constraints
        n = len(point)
        for i, pi in enumerate(point):
            for j in range(i + 1, n):
                pj = point[j]
                difference = pi - pj
                if self.dbm_difference[i][j] < difference:
                    self.dbm_difference[i][j] = difference
                else:
                    difference = - difference
                    if self.dbm_difference[j][i] < difference:
                        self.dbm_difference[j][i] = difference
                sum = pi + pj
                if self.dbm_sum[i][j] < sum:
                    self.dbm_sum[i][j] = sum
                else:
                    sum = -sum
                    if self.dbm_sum[j][i] < sum:
                        self.dbm_sum[j][i] = sum

    @staticmethod
    def _constraints_to_points(left, right, bottom, top, x_minus_y, y_minus_x, x_plus_y, minus_x_plus_y):
        top_right = [x_plus_y - top, top]
        right_top = [right, x_plus_y - right]
        right_bottom = [right, right - x_minus_y]
        bottom_right = [x_minus_y + bottom, bottom]
        bottom_left = [-minus_x_plus_y - bottom, bottom]
        left_bottom = [left, -minus_x_plus_y - left]
        left_top = [left, y_minus_x + left]
        top_left = [top - y_minus_x, top]
        vertices = [top_right, right_top, right_bottom, bottom_right, bottom_left, left_bottom, left_top, top_left]
        return np.array(vertices)

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
            p1 = self.box.low[z]
            p2 = self.box.high[z]
            plot_interval(ax, p1, p2, color, epsilon, epsilon_relative, is_x_dim=y == -1)
            return

        left = self.box.low[x]
        right = self.box.high[x]
        bottom = self.box.low[y]
        top = self.box.high[y]
        x_minus_y = self.dbm_difference[x][y]  # bound from bottom right
        y_minus_x = self.dbm_difference[y][x]  # bound from top left
        x_plus_y = self.dbm_sum[x][y]  # bound from top right
        minus_x_plus_y = self.dbm_sum[y][x]  # bound from bottom left

        if epsilon == 0 or PLOT_NON_EPSILON_SETS:
            points = Octagon._constraints_to_points(left=left, right=right, bottom=bottom, top=top, x_minus_y=x_minus_y,
                                                    y_minus_x=y_minus_x, x_plus_y=x_plus_y,
                                                    minus_x_plus_y=minus_x_plus_y)
            polygon = Polygon(points, closed=True, linewidth=1, edgecolor=color, facecolor="none")
            ax.add_patch(polygon)
        if epsilon == 0:
            return

        if epsilon_relative:
            horizontal_radius = (right - left) / 2.0
            if horizontal_radius > 0:
                left -= horizontal_radius * epsilon
                right += horizontal_radius * epsilon
            vertical_radius = (top - bottom) / 2.0
            if vertical_radius > 0:
                bottom -= vertical_radius * epsilon
                top += vertical_radius * epsilon
            diagonal_radius = (x_minus_y + y_minus_x) / 2.0
            if diagonal_radius > 0:
                x_minus_y += diagonal_radius * epsilon
                y_minus_x += diagonal_radius * epsilon
            diagonal_radius = (x_plus_y + minus_x_plus_y) / 2.0
            if diagonal_radius > 0:
                x_plus_y += diagonal_radius * epsilon
                minus_x_plus_y += diagonal_radius * epsilon
        else:
            left -= epsilon
            right += epsilon
            bottom -= epsilon
            top += epsilon
            x_minus_y += epsilon
            y_minus_x += epsilon
            x_plus_y += epsilon
            minus_x_plus_y += epsilon

        points = Octagon._constraints_to_points(left=left, right=right, bottom=bottom, top=top, x_minus_y=x_minus_y,
                                                y_minus_x=y_minus_x, x_plus_y=x_plus_y, minus_x_plus_y=minus_x_plus_y)
        line_style = "--" if PLOT_NON_EPSILON_SETS else "-"
        polygon = Polygon(points, closed=True, linewidth=1, edgecolor=color, facecolor="none", linestyle=line_style)
        ax.add_patch(polygon)

    def dimension(self):
        return self.box.dimension()

    def half_spaces(self, epsilon, epsilon_relative):
        # chain box constraints and diagonal constraints
        return chain(self.box.half_spaces(epsilon=epsilon, epsilon_relative=epsilon_relative),
                     HalfSpaceIteratorOctagonDiagonals(self, sum=True, epsilon=epsilon,
                                                       epsilon_relative=epsilon_relative),
                     HalfSpaceIteratorOctagonDiagonals(self, sum=False, epsilon=epsilon,
                                                       epsilon_relative=epsilon_relative))


class HalfSpaceIteratorOctagonDiagonals(object):
    def __init__(self, octagon: Octagon, sum: bool, epsilon, epsilon_relative):
        self.octagon = octagon
        self.sum = sum
        self.epsilon = epsilon
        self.epsilon_relative = epsilon_relative
        self.i = 0
        self.j = 1
        self.n = octagon.dimension()

    def __iter__(self):
        return self

    def __next__(self):
        i = self.i
        j = self.j
        if j == i:
            if i == self.n - 1:
                raise StopIteration()
            self.j += 1
            j += 1
        if j == self.n - 1:
            self.i = i + 1
            self.j = 0
        else:
            self.j += 1

        a = [0.0 for _ in range(self.n)]
        dbm = self.octagon.dbm_sum if self.sum else self.octagon.dbm_difference
        if self.sum:
            if i < j:
                a[i] = 1.0
                a[j] = 1.0
            else:
                a[i] = -1.0
                a[j] = -1.0
        else:
            a[i] = 1.0
            a[j] = -1.0

        b = dbm[i][j]
        if self.epsilon > 0.0:
            if self.epsilon_relative:
                b2 = dbm[j][i]
                bloating_distance = (b + b2) / 2.0 * self.epsilon
            else:
                bloating_distance = self.epsilon
            b += bloating_distance
        return a, b
