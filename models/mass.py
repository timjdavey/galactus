import numpy as np
from functools import cached_property
from .space import Space

class Mass:

    def __init__(self, space):
        self.space = space

    def generate(self, func):
        space = self.space
        volume = space.blank()
        c = space.center*space.scale
    
        for ijk in space.list:
            d = ijk*space.scale
            z = np.abs(c[2] - d[2])
            r = ((c[1]-d[1])**2 + (c[0]-d[0])**2)**0.5
            volume[tuple(ijk)] = func(r,z)
        return volume

    @cached_property
    def space(self):
        return Space((self.points, self.points, self.points), self.scale)

    @cached_property
    def masses(self):
        return [self.generate(func) for func in [buldge, thick, thin]]