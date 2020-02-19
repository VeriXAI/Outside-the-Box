from matplotlib.patches import Polygon
from scipy.optimize import linprog
from copy import deepcopy
from numpy import array, dot
# from pypoman import compute_polytope_halfspaces, compute_polytope_vertices

from .PointCollection import PointCollection
from utils import *


class ConvexHull(PointCollection):
    def __init__(self, dimension):
        super().__init__()
        self.points = []
        self.dim = dimension
        self._center = None
        self.A = None
        self.b = None

    def __str__(self):
        if self.isempty():
            return "raw ConvexHull"
        return "  ConvexHull(m={:d})".format(len(self.points))

    def create(self, points, remove_redundancies=CONVEX_HULL_REDUNDANCY_REMOVAL):
        super().create(deepcopy(points[0]))
        for point in points[1:]:
            super().add(point)

        if remove_redundancies:
            self.points = ConvexHull._convex_hull(points)
        else:
            self.points = points

        # convert to H-rep for non-flat sets
        # (however, the number of vertices is only a necessary criterion for not being flat)
        if self._is_corner_case():
            print_flat_convex_hull_warning()
        else:
            self._tohrep()

    def contains(self, point, confidence_fun, bloating=0.0, bloating_relative=False, skip_confidence=False,
                 novelty_mode=False):
        assert bloating >= 0, "bloating must be nonnegative"
        if bloating_relative:
            raise(NotImplementedError("Convex hull does not support relative bloating at the moment."))

        if self.A is None or self.b is None:
            # V-rep
            if not self._is_corner_case():
                print("Warning: Using slow vertex representation. Consider converting to H-representation.")
                if bloating > 0:
                    raise (NotImplementedError(
                        "Convex hull in V-representation does not support bloating at the moment."))
            inside = ConvexHull._inhull(point, self.points)
        else:
            # H-rep
            inside = True
            for Ai, bi in zip(self.A, self.b):
                inside = ConvexHull._in_half_space(Ai, bi, point, bloating)
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
        raise(NotImplementedError())

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
            projections = [p[z] for p in self.points]
            p1 = min(projections)
            p2 = max(projections)
            plot_interval(ax, p1, p2, color, epsilon, epsilon_relative, is_x_dim=y == -1)
            return

        if epsilon == 0:
            points = self.points
        elif epsilon_relative:
            raise NotImplementedError("Convex hull does not support relative bloating at the moment.")
        elif self._is_corner_case():  # no bloating for flat sets available
            points = self.points
        else:
            # convert to H-rep
            if self.A is None or self.b is None:
                self._tohrep()
            # bloat
            b = [bi + epsilon for bi in self.b]
            # convert back to V-rep
            points = compute_polytope_vertices(array(self.A), array(b))

        points = ConvexHull._convex_hull_2d([[point[x], point[y]] for point in points])
        polygon = Polygon(points, closed=True, linewidth=1, edgecolor=color, facecolor="none")
        ax.add_patch(polygon)

    def dimension(self):
        return self.dim

    def center(self):
        if self._center is None:
            self._center = sum(self.points) / len(self.points)
        return self._center

    @staticmethod
    def _inhull(point, points):
        m = len(points)
        if m == 1:
            for pi, qi in zip(point, points[0]):
                if pi != qi:
                    return False
            return True

        n = len(point)
        c = [np.float32(0) for _ in range(m)]
        A = []
        for j in range(m):
            pj = points[j]
            if j == 0:
                for i in range(n):
                    A.append([pj[i]])
            else:
                for i in range(n):
                    A[i].append(pj[i])
        A.append([np.float32(1) for _ in range(m)])
        b = [pj for pj in point]
        b.append(np.float32(1))

        try:
            res = linprog(c, A_eq=A, b_eq=b)
            status = res.status
        except ValueError as e:
            if e.__str__() == "The algorithm terminated successfully and determined that the problem is infeasible.":
                status = 2
            else:
                raise e
        if status == 0:  # solution found
            return True
        elif status == 2:  # infeasible
            return False
        raise (ValueError("LP solver returned status {:d}".format(status)))

    @staticmethod
    def _convex_hull(points: list):
        m = len(points)
        if m < 2:
            return points
        if len(points[0]) == 2:
            return ConvexHull._convex_hull_2d(points)

        # courageously remove k points in one go
        if CONVEX_HULL_REMOVE_BATCHES:
            k = m // 20
            j = m-1
            while j > k:
                removed_points = []
                for i in range(k):
                    removed_points.append(points.pop())
                j -= k
                for point in removed_points:
                    if not ConvexHull._inhull(point, points):
                        points.extend(removed_points)
                        break

        # remove one point at a time
        i = 0
        j = m-1
        while j >= i and j > 0:
            point = points.pop()
            if ConvexHull._inhull(point, points):
                j -= 1
            else:
                if i == j:
                    points.append(point)
                else:
                    points.append(points[i])
                    points[i] = point
                i += 1
        if PRINT_CONVEX_HULL_SAVED_VERTICES:
            print("convex hull saved {:d}/{:d} vertices".format(m - len(points), m))
        return points

    @staticmethod
    def _right_turn(O, u, v):
        return (u[0] - O[0]) * (v[1] - O[1]) - (u[1] - O[1]) * (v[0] - O[0])

    @staticmethod
    def _semihull(iterator, points):
        semihull = []
        for i in iterator:
            while len(semihull) >= 2 and ConvexHull._right_turn(semihull[-2], semihull[-1], points[i]) <= 0:
                semihull.pop()
            semihull.append(points[i])
        return semihull

    @staticmethod
    def _convex_hull_2d(points):
        m = len(points)
        if m < 2:
            return points

        # sort the points lexicographically
        points.sort(key=lambda x: (x[0], x[1]))

        # build lower hull
        lower = ConvexHull._semihull(range(m), points)

        # build upper hull
        upper = ConvexHull._semihull(range(m-1, -1, -1), points)

        # remove the last point of each segment because they are repeated
        new_points = []
        new_points.extend(lower[:-1])
        new_points.extend(upper[:-1])

        if PRINT_CONVEX_HULL_SAVED_VERTICES:
            print("convex hull saved {:d}/{:d} vertices".format(m - len(new_points), m))
        return new_points

    def _tohrep(self):
        self.A, self.b = compute_polytope_halfspaces(self.points)

    @staticmethod
    def _in_half_space(a, b, point, bloating):
        return dot(a, point) <= b + bloating

    def _is_corner_case(self):
        return len(self.points) <= self.dim

    def half_spaces(self, epsilon, epsilon_relative):
        return HalfSpaceIteratorConvexHull(self.A, self.b, epsilon, epsilon_relative)


class HalfSpaceIteratorConvexHull(object):
    def __init__(self, A, b, epsilon, epsilon_relative):
        self.A = A
        self.b = b
        self.epsilon = epsilon
        self.epsilon_relative = epsilon_relative
        if self.epsilon_relative:
            raise NotImplementedError("Relative bloating is not available.")
        self.i = 0
        self.n = 0 if self.b is None else len(self.b)

    def __iter__(self):
        return self

    def __next__(self):
        i = self.i
        if i == self.n:
            raise StopIteration()
        self.i += 1
        return self.A[i], self.b[i] + self.epsilon
