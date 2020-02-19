from .SetBasedAbstraction import SetBasedAbstraction
from .Box import Box


class BoxAbstraction(SetBasedAbstraction):
    def __init__(self, confidence_fun, size=1, epsilon=0., epsilon_relative=True):
        super().__init__(confidence_fun, size, epsilon, epsilon_relative)

    def name(self):
        return "Box"

    def set_type(self):
        return Box
