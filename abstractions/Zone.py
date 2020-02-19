from matplotlib.patches import Polygon

from .PointCollection import PointCollection
from utils import *


"""
Zones are represented by a DBM, which is stored in a row-major way.
An entry 'e = M[i][j]' represents the constraint 'x_i - x_j <= e'.
The last row/column 'n' represent the constant 0, so 'x_i - 0 <= M[i][n]' and '0 - x_j <= M[n][j]'.
"""
class Zone(PointCollection):
    def __init__(self, dimension):
        super().__init__()
        self.dbm = [[] for _ in range(dimension + 1)]

    def __str__(self):
        if self.isempty():
            return "raw Zone"
        return "  Zone(DBM = " + str(self.dbm) + ")\n"

    def create(self, point):
        super().create(point)

        constant_row = self.dbm[-1]
        for i, pi in enumerate(point):
            row = self.dbm[i]
            for j, pj in enumerate(point):
                row.append(pi - pj)  # xi - xj
            row.append(pi)  # xi - 0
            constant_row.append(-pi)  # 0 - xi
        constant_row.append(0)  # 0 - 0

    def contains(self, point, confidence_fun, bloating=0.0, bloating_relative=True, skip_confidence=False,
                 novelty_mode=False):
        assert bloating >= 0, "bloating must be nonnegative"
        n = len(point)
        inside = True
        if bloating_relative:
            for i, pi in enumerate(point):
                for j in range(i + 1, n):
                    pj = point[j]
                    bloating_distance = abs(self.dbm[i][j] - self.dbm[j][i]) / 2.0 * bloating
                    if not (-self.dbm[j][i] - bloating_distance <= pi - pj <= self.dbm[i][j] + bloating_distance):
                        inside = False
                        break
                if not inside:
                    break
                bloating_distance = abs(self.dbm[i][n] - self.dbm[n][i]) / 2.0 * bloating
                if not (-self.dbm[n][i] - bloating_distance <= pi <= self.dbm[i][n] + bloating_distance):
                    inside = False
                    break
        else:
            for i, pi in enumerate(point):
                for j, pj in enumerate(point):
                    if not (pi - pj <= self.dbm[i][j] + bloating):
                        inside = False
                        break
                if not inside:
                    break
                if not (-self.dbm[n][i] - bloating <= pi <= self.dbm[i][n] + bloating):
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

        n = len(point)
        for i, pi in enumerate(point):
            for j in range(i + 1, n):
                pj = point[j]
                difference = pi - pj
                if self.dbm[i][j] < difference:
                    self.dbm[i][j] = difference
                else:
                    difference = - difference
                    if self.dbm[j][i] < difference:
                        self.dbm[j][i] = difference
            if self.dbm[i][n] < pi:
                self.dbm[i][n] = pi
            elif self.dbm[n][i] < -pi:
                self.dbm[n][i] = -pi

    @staticmethod
    def _constraints_to_points(left, right, bottom, top, x_minus_y, y_minus_x):
        top_right = [right, top]
        rb_top = [right, right - x_minus_y]
        rb_bottom = [x_minus_y + bottom, bottom]
        bottom_left = [left, bottom]
        lt_bottom = [left, y_minus_x + left]
        lt_top = [top - y_minus_x, top]
        return np.array([top_right, rb_top, rb_bottom, bottom_left, lt_bottom, lt_top])

    def plot(self, dims, color, epsilon, epsilon_relative, ax):
        x = dims[0]
        y = dims[1]
        n = self.dimension()
        if x == -1 and y == -1:
            plot_zero_point(ax, color, epsilon, epsilon_relative)
            return
        elif x == -1 or y == -1:
            if x == -1:
                z = y
            else:
                z = x
            p1 = -self.dbm[n][z]
            p2 = self.dbm[z][n]
            plot_interval(ax, p1, p2, color, epsilon, epsilon_relative, is_x_dim=y == -1)
            return

        left = -self.dbm[n][x]
        right = self.dbm[x][n]
        bottom = -self.dbm[n][y]
        top = self.dbm[y][n]
        x_minus_y = self.dbm[x][y]  # bound from bottom right
        y_minus_x = self.dbm[y][x]  # bound from top left

        if epsilon == 0 or PLOT_NON_EPSILON_SETS:
            points = Zone._constraints_to_points(left=left, right=right, bottom=bottom, top=top, x_minus_y=x_minus_y,
                                                 y_minus_x=y_minus_x)
            polygon = Polygon(points, closed=True, linewidth=1, edgecolor=color, facecolor="none")
            ax.add_patch(polygon)
        if epsilon == 0:
            return

        if epsilon_relative:
            width = (right - left) / 2.0
            if width > 0:
                left -= width * epsilon
                right += width * epsilon
            height = (top - bottom) / 2.0
            if height > 0:
                bottom -= height * epsilon
                top += height * epsilon
            diagonal_radius = (x_minus_y + y_minus_x) / 2.0
            if diagonal_radius > 0:
                x_minus_y += diagonal_radius * epsilon
                y_minus_x += diagonal_radius * epsilon
        else:
            left -= epsilon
            right += epsilon
            bottom -= epsilon
            top += epsilon
            x_minus_y += epsilon
            y_minus_x += epsilon

        points = Zone._constraints_to_points(left=left, right=right, bottom=bottom, top=top, x_minus_y=x_minus_y,
                                             y_minus_x=y_minus_x)
        line_style = "--" if PLOT_NON_EPSILON_SETS else "-"
        zone_eps = Polygon(points, closed=True, linewidth=1, edgecolor=color, facecolor="none", linestyle=line_style)
        ax.add_patch(zone_eps)

    def dimension(self):
        return len(self.dbm) - 1

    def half_spaces(self, epsilon, epsilon_relative):
        return HalfSpaceIteratorZone(self, epsilon, epsilon_relative)


class HalfSpaceIteratorZone(object):
    def __init__(self, zone: Zone, epsilon, epsilon_relative):
        self.zone = zone
        self.epsilon = epsilon
        self.epsilon_relative = epsilon_relative
        self.i = 0
        self.j = 1
        self.n = zone.dimension()

    def __iter__(self):
        return self

    def __next__(self):
        i = self.i
        j = self.j
        if j == i:
            if i == self.n:
                raise StopIteration()
            self.j += 1
            j += 1
        if j == self.n:
            self.i = i + 1
            self.j = 0
        else:
            self.j += 1

        a = [0.0 for _ in range(self.n)]
        if j == self.n:
            a[i] = 1.0
        elif i == self.n:
            a[j] = -1.0
        else:
            a[i] = 1.0
            a[j] = -1.0

        v = self.zone.dbm[i][j]
        if self.epsilon_relative:
            bloating_distance = abs(v - self.zone.dbm[j][i]) / 2.0 * self.epsilon
        else:
            bloating_distance = self.epsilon
        b = v + bloating_distance
        return a, b
