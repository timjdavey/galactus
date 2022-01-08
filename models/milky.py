import numpy as np
from functools import cached_property
from .space import Space
from .simulation import Simulation

class MilkyWay(Simulation):
    """
    Wrapper around Simulation.

    Given a certain number of grid `points`,
    returns a `Simulation` with MilkyWay mass profile.
    
    Equations below from
    https://academic.oup.com/mnras/article/414/3/2446/1042117?login=true#m1

    With specific values from
    https://academic.oup.com/view-large/18663759
    """
    def __init__(self, points, radius=1, cp=None, *args, **kwargs):
        self.points = points
        self.radius = radius
        self.scale = radius*2/points
        self.cp = cp

        self.log('gen space')
        space = Space((self.points, self.points, self.points), self.scale)

        masses = []
        c = space.center*space.scale
        t = self.points**3
        tmod = int(t/100)
        for func in [buldge, thick, thin]:
            volume = space.blank()
            
            for i, ijk in enumerate(space.list):
                if i % tmod == 0:
                    self.log('gen %s %s' % (func.__name__, i/t))

                d = ijk*space.scale
                z = np.abs(c[0] - d[0])
                r = ((c[1]-d[1])**2 + (c[2]-d[2])**2)**0.5
                volume[tuple(ijk)] = func(r,z)
            
            masses.append(volume)

        self.log()
        super().__init__(masses, space, cp=cp, *args, **kwargs)


def buldge(R, z,
        p0=9.93, # *10^10 2.1 scale density M kpi^-3 
        q=0.5, # 2.1 axial ratio
        rcut=2.1, # kpc 2.1
        r0=0.075, # kpc 2.1
        alpha=1.8): # 2.1
    rprime = (R**2+(z/q)**2)**0.5
    exponent = np.exp(-(rprime/rcut)**2)
    return p0*exponent/((1+rprime/r0)**alpha)

def disk(R, z, zd, sig0, Rd):
    exponent = np.exp((-np.abs(z)/zd)-(R/Rd))
    return sig0*exponent/(2*zd)
    
def thick(R,z):
    return disk(R, z,
        zd=0.9, # kpc Table 1
        sig0=209.5, # Table 2
        Rd=3.31) # Table 2 (T1 has value of 3.6)

def thin(R,z):
    return disk(R, z,
        zd=0.3, # kpc Table 1
        sig0=816.6, # Table 2
        Rd=2.9) # Table 2 (T1 has value of 2.6)

