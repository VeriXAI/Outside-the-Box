from .SetBasedAbstraction import SetBasedAbstraction
from .MeanBall import MeanBall


class MeanBallAbstraction(SetBasedAbstraction):
    def __init__(self, confidence_fun, size=1, epsilon=0., epsilon_relative=True):
        super().__init__(confidence_fun, size, epsilon, epsilon_relative)

    def name(self):
        return "MeanBall"

    def set_type(self):
        return MeanBall

    def add_clustered_to_set(self, values, cj, mean_computer):
        if len(values) == 0:
            return
        _set = self.sets[cj]  # type: MeanBall
        assert _set.isempty()
        center = mean_computer()
        _set.create(values[0], center=center)
        for vj in values[1:]:
            _set.add(vj)
