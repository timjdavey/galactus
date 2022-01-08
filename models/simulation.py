import numpy as np
import pandas as pd
import seaborn as sns
from functools import cached_property
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from .space import Space



class Simulation:
    """
    Main simulation object
    """
    def __init__(self, masses, space, G=1, cp=None):
        masses = np.array(masses)
        for i, mass in enumerate(masses):
            if mass.shape != space.points:
                raise ValueError("mass %s shape does not match space grid (%s, %s)" % (i, mass.shape, space.points))

        self.mass_components = masses
        self.space = space
        self.G = G
        self.cp = cp

        self.sums = {'mass': np.sum(masses, axis=0)}

    def analyse(self, sub_list=None, imod=100):
        """
        Does the main bulk of the analysis

        :sub_list: array_like of points to calculate for, rather than whole of space
        """

        # first assumption is all masses
        # in each grid point
        # are summed to a single mass for that grid
        # so densities are treated normally
        space = self.space
        sums = self.sums
        mass_sum = sums['mass']
        
        point_list = space.list if sub_list is None else sub_list
    
        # then calculates the affect of the mass in
        # each grid point on all other grid points
    
        # we approximate the mass within a given grid
        # to be the center of mass for the grid
        # but outside of each grid, we never make that assumption
    
        fields = dict([(d, space.blank()) for d in space.dimensions])
        for d in space.dimensions:
            fields['a%s' % d] = space.blank()
        fields['abs'] = space.blank()
    
        # for each point in space impacting others,
        # work out the vector & scalar force
        # observed for each other point in space
    
        # only need to do for points with nonzero mass
        nonzero_masses = np.nonzero(mass_sum)
        total_mass_points = len(nonzero_masses[0])
        for i in range(total_mass_points):
            mass_points = tuple([nonzero_masses[d][i] for d in space.dimensions])
            
            if i % imod == 0:
                self.log("%s percent analysed" % (i/total_mass_points))
            
            M = mass_sum[mass_points]
            mass_distances = np.array(mass_points)*space.scale
    
            for observed_points in point_list:
        
                if np.array_equal(mass_points,observed_points):
                # the mass of a certain grid point
                # has no impact on it's own grid point
                # to avoid infinities etc
                # so leave as 0
                    pass
                else:
                    deltas = np.array(observed_points)*space.scale-mass_distances
        
                    r2 = np.sum([d**2 for d in deltas])
                    r = r2**0.5
        
                    F = self.G*M/r2
        
                    ijk = tuple(observed_points)
                    for i, delta in enumerate(deltas):
                        g = -F*delta/r
                        fields[i][ijk] += g
                        fields['a%s' % i][ijk] += np.abs(g)
        
                    fields['abs'][ijk] += np.abs(F)
    
        self.log('sum')
        fields['mass'] = sums['mass']
        self.sums = fields
        self._sums_original = sums.copy()
        self.log()
    
    def analyse_radius(self, imod=100):
        """ Analyses just for the radius """
        self.analyse(self.space.radius_list, imod)

    def log(self, *args, **kwargs):
        """ Outputs to notebook """
        if self.cp is not None: self.cp(*args, **kwargs)

    def rotate(self, d1, d2):
        """
        Rotates the sums & space objects, not the components
        :d1: (0,1,2) or ('z','y','x')
        :d2: second dimension to swap
        """

        # convert to dimension int to reference
        dims = dict([(key,i) for i, key in enumerate(['z','y','x'])])
        if d1 in dims.keys() and d2 in dims.keys():
            d1, d2 = dims[d1],dims[d2]
        elif d1 not in dims.values() or d2 not in dims.values():
            raise ValueError('Invalid dimensions')

        # sums
        dic = self.sums
        self._sums = dic.copy()
        for key, value in dic.items():
            self.log(key)
            dic[key] = np.rot90(dic[key], axes=(d1,d2))
        self.sums = dic


        # space
        self.log('space')
        self._space = self.space
        old_points = self.space.points
        new_points = list(old_points)
        new_points[d2] = old_points[d1]
        new_points[d1] = old_points[d2]
        self.space = Space(new_points, self.space.scale)

        self.log()

    def reset_rotation(self):
        """ Resets the simulation back to it's original orientation """
        try:
            self.space = self._space
            self.sums = self._sums
        except AttributeError:
            pass # if no rotation has happened
    
    def _heatmap(self, grids, col, axes, cmap, title=None):
        self.log(title)
        vmin, vmax = grids.min(), grids.max()
        lm = len(grids)
        
        for i, grid in enumerate(grids):
            f = sns.heatmap(grid, ax=axes[i][col], vmin=vmin, vmax=vmax,
                cbar= (i == 0), cmap=cmap)
            if i == 0:
                f.set(title=title)
        self.log()

    def heatmaps(self, slices=None, image_scale=15):
        """
        Plots all the heatmaps of all the sums data (masses & fields),
        for all z axis stacks.

        :slices: allows you to show a reduced number (from, to, how many)
        """
        if slices is None:
            how_many = self.space.points[0]
        else:
            how_many = len(self.sums['mass'][slices[0]:slices[1]:slices[2]])
        
        squares = len(self.sums)
        fig, axes = plt.subplots(how_many, squares,
            figsize=(image_scale,how_many*image_scale/squares))

        for i, k in enumerate(self.sums.keys()):
            if k in ('abs','mass'):
                cmap, title = 'cubehelix', k
            else:
                cmap, title = 'icefire', 'dim %s' % k

            if slices is None:
                grids = self.sums[k] 
            else:
                grids = self.sums[k][slices[0]:slices[1]:slices[2]]

            self._heatmap(grids, i, axes, cmap=cmap, title=title)

    def lines(self, stack=None, row=None):
        """ Plots lines of an x axis row for all sums """
        if stack is None: stack = self.space.center[0]
        if row is None: row = self.space.center[1]

        fig, axes = plt.subplots(1, len(self.sums), figsize=(15,5))
        for i, k in enumerate(self.sums.keys()):
            g = sns.lineplot(y=self.sums[k][stack][row], x=self.space.x, ax=axes[i])
            g.set(title=k)

    def mass_profile(self, stack=None, cmap='Spectral_r'):
        """ Heatmap of mass over space """
        if stack is None: stack = self.space.center[0]
        fig, axes = plt.subplots(figsize=(15,10))
        sns.heatmap(self.sums['mass'][stack],
            cmap=cmap,
            square=True, norm=LogNorm(),
            xticklabels=self.space.x, yticklabels=self.space.coords[1]*self.space.scale)








