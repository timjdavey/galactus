import numpy as np
from functools import cached_property

class Space:
    """
    Simple class for handling the common features of
    coordinate systems etc, for N dimensions.
    """
    def __init__(self, points, scale=1):
        self.points = points
        self.scale = scale

    def blank(self):
        return np.zeros(self.points)

    @cached_property
    def dimensions(self):
        return range(len(self.points))

    @cached_property
    def center(self):
        return np.array([int(p/2) for p in self.points])

    @cached_property
    def coords(self):
        return [np.array(range(self.points[d])) for d in self.dimensions]

    @cached_property
    def x(self):
        return self.coords[-1]*self.scale

    @cached_property
    def list(self):
        return np.array(np.meshgrid(*self.coords)).T.reshape(-1,len(self.points))
