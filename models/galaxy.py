import numpy as np
import time
from multiprocessing import Pool, cpu_count
from functools import cached_property, partial
from .space import Space
from .simulation import Simulation
from .memory import memory_usage
from models.equations import velocity

SCALAR_SOLAR = 8500 # kms-1
TAU = 0.00000037 # kms-1

class Galaxy(Simulation):
    """
    Makes the Simulation model into a disk
    """

    def radius_points(self, radius=None, points=None):
        """
        Calculates for a set number of :points:, up to a maximum :radius:
        Returns a list of points to analyse
        """
        calc_radius = radius if radius is not None else self.radius
        percent = radius/self.space.radius
        rl = self.space.radius_list
        max_point = len(rl)*percent
        calc_points = points if points is not None else max_point
        return rl[:int(max_point)+1:max(int(max_point/points),1)][:points]

    def dataframe(self, *args, **kwargs):
        """
        Returns analysis as a dataframe, adding the radius
        Optional `R` so can scale from kpc to m
        """
        df = super().dataframe(*args, **kwargs)
        c = self.space.center
        scale = self.space.scale
        df['zd'] = (df['z']-c[0])*scale
        #if R is None: R = 1
        df['rd'] = (scale*((df['y']-c[1])**2 + (df['x']-c[2])**2)**0.5)
        return df

    def get_velocities(self, R=None):
        """ Gets the velocities for a given set of data points """
        # if want more accurate can just do without the +1 as well
        # and when creating in `scalar_fit` rotmass_points(space, left=True)
        cdf = self.dataframe()
        if R is None: R = self.profile.rotmass_df['R']
        return velocity(R, np.interp(R, cdf['rd'], cdf['x_vec']))

    def smog_convert(self, tau=TAU, reference_scalar=SCALAR_SOLAR, analyse=True):
        """
        For a given scalar map galaxy,
        generates a new Galaxy with the calculated at calculated `points`
        """
        new_masses = self.mass_components*np.sqrt(1+reference_scalar/tau)/np.sqrt(1+self.scalar_map()/tau)
        new_galaxy = Galaxy(new_masses, self.space, mass_labels=self.mass_labels, cp=self.cp)
        if hasattr(self, 'profile'): new_galaxy.profile = self.profile
        return new_galaxy



