import numpy as np
from functools import cached_property
from .space import Space
from .simulation import Simulation

class MilkyWay(Simulation):
    """
    Wrapper around Simulation.

    Given a certain number of grid `points`,
    returns a `Simulation` with MilkyWay mass profile.
    """
    def __init__(self, points, radius=1, cp=None, *args, **kwargs):
        self.points = points
        self.radius = radius
        self.scale = radius*2/points
        self.cp = cp

        self.log('gen space')
        space = Space((self.points, self.points, self.points), self.scale)

        masses = []
        for func in [buldge, thick, thin]:
            self.log('gen %s' % func.__name__)
            masses.append(generate_masses(space, func))

        self.log()
        super().__init__(masses, space, *args, **kwargs)



def generate_masses(space, func):
    volume = space.blank()
    c = space.center*space.scale
    
    for ijk in space.list:
        d = ijk*space.scale
        z = np.abs(c[0] - d[0])
        r = ((c[1]-d[1])**2 + (c[2]-d[2])**2)**0.5
        volume[tuple(ijk)] = func(r,z)
    return volume

def buldge(R, z, p0=9.93, q=0.5, rcut=2.1, r0=0.075, alpha=1.8):
    rprime = (R**2+(z/q)**2)**0.5
    exponent = np.exp(-(rprime/rcut)**2)
    return p0*exponent/((1+rprime/r0)**alpha)

def disk(R, z, zd, sig0, Rd):
    exponent = np.exp((-np.abs(z)/zd)-(R/Rd))
    return sig0*exponent/(2*zd)
    
def thick(R,z):
    return disk(R, z, zd=900, sig0=209.5, Rd=3.31)

def thin(R,z):
    return disk(R, z, zd=300, sig0=816.6, Rd=2.9)

