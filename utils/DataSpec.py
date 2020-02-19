from . import *


class DataSpec(object):
    def __init__(self, file=None, n=None, randomize=False, classes=None, x=None, y=None):
        self.file = file
        self.n = n
        self.randomize = randomize
        self.classes = classes
        self._x = x
        self._y = y

    def x(self):
        return self._x

    def y(self):
        return self._y

    def ground_truths(self):
        return categoricals2numbers(self._y)

    def set_data(self, data=None, x=None, y=None):
        if data is not None:
            # extract data from dict
            self.set_x(data["features"])
            self.set_y(data["labels"])
        else:
            self.set_x(x)
            self.set_y(y)

    def set_x(self, x):
        self._x = x

    def set_y(self, y):
        self._y = y

    def has_data(self):
        return self._x is not None and self._y is not None

    def filter(self, filter):
        self._x = self._x[filter]
        self._y = self._y[filter]
        self.n = len(self._y)
