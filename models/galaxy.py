import numpy as np
import time
from multiprocessing import Pool, cpu_count
from functools import cached_property, partial
from .space import Space
from .simulation import Simulation


def rz(ijk, space_scale, c):
    """
    Calculates r & z.
    """
    d = ijk*space_scale
    z = np.abs(c[0] - d[0])
    r = ((c[1]-d[1])**2 + (c[2]-d[2])**2)**0.5
    return r,z
    

def mass_worker(ijk, funcs, space_scale, vol_per_grid, c, combine):
    """
    Calculates the mass for a given point in space.

    :combine: combines all the masses into a single mass.
    """
    ijk = np.array(ijk)
    if combine:
        density = 0
        for func in funcs:
            density += func(*rz(ijk, space_scale, c))
        return (ijk, [density*vol_per_grid,])
    else:
        masses = [func(*rz(ijk, space_scale, c))*vol_per_grid for func in funcs]
        return (ijk, masses)


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
    def __init__(self, profiles, points, radius=1, zcut=5, multi=None, cp=None, combine_masses=False, *args, **kwargs):
        self.points = points
        self.radius = radius
        self.scale = radius*2/points
        self.cp = cp
        self.profiles = profiles

        self.log('gen space')
        space = Space((int(self.points/zcut), self.points, self.points), self.scale)

        fl = dict([(f.__name__, f) for f in (buldge, disk)])
        funcs = [partial(fl[p['func']], **p['params']) for p in profiles.values()]

        masses = []
        for i in range(len(funcs)):
            masses.append(space.blank())
        #masses = np.array(masses)

        worker = partial(mass_worker,
            funcs=funcs,
            space_scale=space.scale,
            vol_per_grid=space.scale**3,
            c=space.center*space.scale,
            combine=combine_masses)
        mass_labels = ['combined',] if combine_masses else list(profiles.keys())

        tasks = space.count
        self.log('gen masses for %s points' % tasks)
        tic = time.perf_counter()
        chunksize = max(1, tasks//(cpu_count()*(2**11)))
        every = tasks/10
        
        with Pool() as pl:
            count = 0
            for ijk, mres in pl.imap_unordered(worker, space.list, chunksize=chunksize):
                if count % every == 0: self.log("%s%%" % (count*100/tasks))
                count += 1
                tp = tuple(ijk)
                for i, m in enumerate(mres):
                    masses[i][tp] = m

        toc = time.perf_counter()
        self.log("completed in %s seconds" % (toc-tic))
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
        return rl[:int(max_point)+1:max(int(max_point/points),1)]

    @cached_property
    def dataframe(self):
        """ Returns analysis as a dataframe, adding the radius """
        df = super().dataframe
        space_scale = self.space.scale
        c = self.space.center*space_scale
        
        rs, zs = [], []
        for i, ijk in df['ijk'].items():
            r,z = rz(np.array(ijk), space_scale, c)
            rs.append(r)
            zs.append(z)
        df['rs'] = rs
        df['zs'] = zs
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
    if R > 0 and Rhole > 0: expo -= Rhole/R
    return sig0*np.exp(expo)/(2*zd)

