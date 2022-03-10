import numpy as np
import time
from multiprocessing import Pool, cpu_count
from functools import cached_property, partial
from .space import Space
from .simulation import Simulation
from .memory import memory_usage

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
    def __init__(self, profiles, points, radius=1, zcut=5, cp=None, *args, **kwargs):
        self.points = points
        self.radius = radius
        self.scale = radius*2/points
        self.cp = cp
        self.profiles = profiles

        self.log('gen space, using %s' % memory_usage())
        space = Space((int(self.points/zcut), self.points, self.points), self.scale)

        fl = dict([(f.__name__, f) for f in (buldge, disk)])

        tic = time.perf_counter()
        masses = []
        mass_labels = []
        self.log('gen rz, using %s' % memory_usage())
        
        r, z = space.rz()

        for label, p in profiles.items():
            self.log('gen %s, using %s' % (label, memory_usage()))
            # density * volume_per_grid
            masses.append(fl[p['func']](r,z, **p['params'])*(space.scale**3))
            mass_labels.append(label)

        toc = time.perf_counter()
        self.log("completed in %.2fs, using %s" % ((toc-tic), memory_usage()))
        masses = np.array(masses)
        masses.flags.writeable = False
        super().__init__(masses, space, cp=cp, mass_labels=mass_labels, *args, **kwargs)

    def radius_points(self, radius=None, points=None):
        """
        Calculates for a set number of :points:, up to a maximum :radius:
        Returns a list of points to analyse
        """
        calc_radius = radius if radius is not None else self.radius
        percent = radius/self.radius
        rl = self.space.radius_list
        max_point = len(rl)*percent
        calc_points = points if points is not None else max_point
        return rl[:int(max_point)+1:max(int(max_point/points),1)][:points]

    def dataframe(self, *args, **kwargs):
        """ Returns analysis as a dataframe, adding the radius """
        df = super().dataframe(*args, **kwargs)
        c = self.space.center
        scale = self.space.scale
        df['zd'] = (df['z']-c[0])*scale
        df['rd'] = scale*((df['y']-c[1])**2 + (df['x']-c[2])**2)**0.5
        return df

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
    if Rhole > 0:
        # to account for divide by zero
        rcenter = R.shape[1]//2
        R[:,rcenter,rcenter] = 1e-6
        expo -= Rhole/R
    return sig0*np.exp(expo)/(2*zd)

