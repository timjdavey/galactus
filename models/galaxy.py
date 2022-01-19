import numpy as np
from functools import partial
from multiprocessing import Pool
from .space import Space
from .simulation import Simulation

def mass_worker(func, volume, space_list, space_scale, c, cp):

    if cp is not None: cp("Start %s" %  func.func.__name__)

    for ijk in space_list:

        d = ijk*space_scale
        z = np.abs(c[0] - d[0])
        r = ((c[1]-d[1])**2 + (c[2]-d[2])**2)**0.5
        density = func(r,z)
        volume[tuple(ijk)] = density*space_scale # mass within grid square
    
    if cp is not None: cp("Finished %s" %  func.func.__name__)

    return volume


class Galaxy(Simulation):
    """
    Wrapper around Simulation.

    Given a certain number of grid `points`,
    returns a `Simulation` with MilkyWay mass profile.
    
    Equations below from
    https://academic.oup.com/mnras/article/414/3/2446/1042117?login=true#m1

    With specific values from
    https://academic.oup.com/view-large/18663759

    Velocity rotation curve
    https://iopscience.iop.org/article/10.3847/1538-4357/aaf648/pdf (table 1)

    """
    def __init__(self, profiles, points, radius=1, cp=None, *args, **kwargs):
        self.points = points
        self.radius = radius
        self.scale = radius*2/points
        self.cp = cp
        self.profiles = profiles

        self.log('gen space')
        space = Space((self.points, self.points, self.points), self.scale)

        worker = partial(mass_worker,
            volume=space.blank(),
            space_list=space.list,
            space_scale=space.scale,
            c=space.center*space.scale,
            cp=self.cp)

        fl = dict([(f.__name__, f) for f in (buldge, disk)])
        generators = [partial(fl[p['func']], **p['params']) for p in profiles.values()]

        self.log('gen masses')
        masses = Pool().map(worker, generators)

        self.log()
        super().__init__(masses, space, cp=cp, mass_labels=list(profiles.keys()), *args, **kwargs)


def buldge(R, z, p0, q, rcut, r0, alpha):
    """
    Equation 1 & 2
    https://academic.oup.com/mnras/article/414/3/2446/1042117?login=true#m1
    """
    rprime = (R**2+(z/q)**2)**0.5
    expo = (rprime/rcut)**2
    denom = 1+(rprime/r0)**alpha
    return p0*np.exp(-expo)/denom

def disk(R, z, zd, sig0, Rd, Rhole=0):
    """
    Equation 3 from
    https://academic.oup.com/mnras/article/414/3/2446/1042117?login=true#m1

    Plus Rhole from
    https://arxiv.org/pdf/1604.01216.pdf (eq 12)
    """
    expo = -(np.abs(z)/zd)-(R/Rd)
    if R > 0 and Rhole > 0: expo -= Rhole/R
    return sig0*np.exp(expo)/(2*zd)

def disk_mass(sig0, Rd):
    """
    Section 2.2
    https://academic.oup.com/mnras/article/414/3/2446/1042117?login=true#m1
    """
    return 2*np.pi*sig0*(Rd**2)

