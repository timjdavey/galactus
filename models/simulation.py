import copy
import time
import numpy as np
import pandas as pd
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
from functools import cached_property, partial
from multiprocessing import Pool, cpu_count
from matplotlib.colors import LogNorm
from .space import Space


G = 6.67430*(10**-11)


def gravity_worker(position, masses, scale):
    indices = np.indices(masses.shape[1:])
    deltas = np.array([(indices[i]-c)*scale for i, c in enumerate(position)])
    r2 = np.sum(deltas**2, axis=0)
    r2[tuple(position)] = 1e6 # handle the divide by zero error for position
    r3 = r2**1.5
    results = []
    for mass in masses:
        F_norm = -mass*deltas/r3
        F_vec = [np.sum(arr) for arr in F_norm]
        F_abs = [np.sum(arr) for arr in np.abs(F_norm)]
        results.append([F_vec, F_abs])
    return position, results


class Simulation:
    """
    Main simulation object
    """
    def __init__(self, masses, space, mass_labels=None, G=G, cp=None):
        masses = np.array(masses)
        masses.flags.writeable = False # makes masses a constant
        self.mass_components = masses
        self.mass_labels = mass_labels
        self.space = space
        self.G = G
        self.cp = cp
        self.results = {}

    def analyse(self, sub_list=None, processes=None):
        """
        Does the main bulk of the analysis

        :sub_list: array_like of points to calculate for, rather than whole of space
        :min_mass: 0, only calculate for points with mass greater than this value

        first assumption is all masses
        in each grid point are summed to a single mass for that grid
        so densities are treated normally

        then calculates the affect of the mass in
        each grid point on all other grid points
    
        we approximate the mass within a given grid
        to be the center of mass for the grid
        but outside of each grid, we never make that assumption

        for each point in space impacting others,
        work out the vector & scalar force
        observed for each other point in space

        only need to do for points with nonzero mass
        """
        tic = time.perf_counter()
        point_list = self.space.list if sub_list is None else sub_list
        tasks = len(point_list)
        worker = partial(gravity_worker,
            masses=self.mass_components,
            scale=self.space.scale)
        if processes is None: processes = cpu_count()
        self.log("Setting up %s gravity tasks with %s processes" % (tasks, processes))
        with Pool(processes) as pl:
            for count, r in enumerate(pl.imap_unordered(worker, point_list, chunksize=1)):
                toc = time.perf_counter()
                self.log("%s of %s, %s seconds left" % (count+1, tasks, (toc-tic)*(tasks-count)/(count+1)))
                self.results[r[0]] = r[1]
        
        toc = time.perf_counter()
        self.log("completed in %s seconds" % (toc-tic))
        self.__clear_cache()
        
    def __clear_cache(self):
        # clear cached_property of dataframe
        try:
            del self.dataframe
        except AttributeError:
            pass

    @cached_property
    def dataframe(self):
        data = []
        dimensions = ('z','y','x')
        for ijk, result in self.results.items():
            sums = {'component': 'sum'}
            for di, d in enumerate(dimensions):
                sums[d] = ijk[di]
            for mi, mass_res in enumerate(result):
                rr = sums.copy()
                rr['component'] = self.mass_labels[mi] if self.mass_labels is not None else mi
                
                for ci, clabel in enumerate(('vec', 'abs')):
                    for di, v in enumerate(mass_res[ci]):
                        key = "%s_%s" % (dimensions[di], clabel)
                        rr[key] = v
                    
                        if mi == 0: sums[key] = v
                        else: sums[key] += v
                
                data.append(rr)
            data.append(sums)
        return pd.DataFrame(data)

    @cached_property
    def mass_sum(self):
        return np.sum(self.mass_components, axis=0)

    def combine_masses(self):
        """ Combines the mass components into a single mass, to speed up analysis of large spaces """
        self.mass_components = np.array([self.mass_sum,])
        self.mass_labels = ['combined',]
        self.results = {}
        self.__clear_cache()

    def log(self, *args, **kwargs):
        """ Outputs to notebook """
        if self.cp is not None: self.cp(*args, **kwargs)

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
        sns.heatmap(self.mass_sum[stack],
            cmap=cmap,
            square=True, norm=LogNorm(),
            xticklabels=self.space.x, yticklabels=self.space.coords[1]*self.space.scale)

    def save(self, filename):
        import pickle
        self.log("Saving %s" % filename)
        with open('%s.pickle' % filename, 'wb') as fh:
            pickle.dump(self, fh)
        self.log("Complete")






