import copy
import time
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
from functools import cached_property
from multiprocessing import Pool, cpu_count
from matplotlib.colors import LogNorm
from .space import Space
from .memory import memory_usage

Gkg = 6.67430*(10**-11) # m3 kg-1 s-2
Gsolar = 4.30091*(10**-6) # kpc Ms-1 (km/s)2


global_masses, global_scale = None, None

def initializer(init_masses, init_scale):
    global global_masses
    global global_scale
    global_masses, global_scale = init_masses, init_scale

def gravity_wrapper(position):
    return gravity_worker(position, global_masses, global_scale)

def gravity_worker(position, masses, scale):
    # matrix of distances in indices space from position
    indices = np.indices(masses.shape[1:])
    r_vec = np.array([(indices[i]-c)*scale for i, c in enumerate(position)])
    
    # |r|^2 square the norm
    # convert that to r^3 to normalise vectors in each axis
    r3 = np.sum(r_vec**2, axis=0)**1.5
    # handle the divide by zero error for it's current position
    try:
        r3[tuple(position)] = 1e6
    except IndexError:
        pass # if out of bounds

    results = []
    for mass in masses:
        F_comp = -mass*r_vec/r3
        # creates F_vec for z,y,x (or flexible num of dimensions)
        F_vec = [np.sum(arr) for arr in F_comp] # np.sum(np.sum(np.sum(F_norm, axis=1), axis=1), axis=1)
        F_scalar = np.sum(np.linalg.norm(F_comp, axis=0))
        Potential = np.sum(np.linalg.norm(F_comp*r_vec, axis=0))
        results.append([F_vec, F_scalar, Potential])
    return (position, results)


class Simulation:
    """
    Main simulation object
    """
    dimensions = ('z','y','x')

    def __init__(self, masses, space, mass_labels=None, G=Gsolar, cp=None):
        if isinstance(masses, list): masses = np.array(masses)
        masses.flags.writeable = False # makes masses a constant
        self.mass_components = masses
        self.mass_sums = [np.sum(m) for m in self.mass_components]
        if mass_labels is not None:
            self.mass_labels = mass_labels
        else:
            self.mass_labels = ["mass %s" % i for i in range(len(masses))]
        self.space = space
        self.G = G
        self.cp = cp
        self.results = {}
        self.fit_ratios = {}

    def analyse(self, sub_list=None, verbose=True, processes=8):
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
        tasks = self.space.count if sub_list is None else len(sub_list)
        count = 0

        # if doing the whole space, use chunksize, otherwise keep it lean and mean
        chunksize = tasks//(processes**2) + 1

        with Pool(processes, initializer, (self.mass_components, self.space.scale)) as pool:
            for p, r in pool.imap_unordered(gravity_wrapper, point_list, chunksize):
                if verbose: self.log("%s of %s completed %s" % (count, tasks, memory_usage()))
                self.results[p] = r
                count += 1
        
        toc = time.perf_counter()
        self.log("completed in %.2f seconds" % (toc-tic))
    
    @cached_property
    def scalar_map(self):
        return self.surface_map(1)

    @cached_property
    def potential_map(self):
        return self.surface_map(2)

    def surface_map(self, index, mass_component=0):
        smap = self.space.blank()
        for point, vals in self.results.items():
            smap[point] = vals[mass_component][index]
        return smap

    def dataframe(self, mass_ratios=False, G=None, combined=False):
        # So can override G
        if G is None:
            G = self.G
        elif G is False:
            G = 1

        # So can override mass_ratios
        # both applied later on dataform creation
        if mass_ratios is False:
            mass_ratios = dict([(c, 1) for c in self.mass_labels])

        # if it's been fit to a velocity curve or total mass ratio
        for key, val in self.fit_ratios.items():
            mass_ratios[key] *= val

        data = []
        
        for ijk, result in self.results.items():
            rr = {}
            for mi, mass_res in enumerate(result):
                component = self.mass_labels[mi]
                rr = dict([(d, ijk[di]) for di, d in enumerate(self.dimensions)])
                rr['component'] = component
                
                # for each of the masses
                for di, v in enumerate(mass_res[0]):
                    rr["%s_vec" % self.dimensions[di]] = v*G*mass_ratios[component]

                rr['F_scalar'] = mass_res[1]*G*mass_ratios[component]
                if len(mass_res) > 2: rr['G_potential'] = mass_res[2]*G*mass_ratios[component]
                data.append(rr)
        
        df = pd.DataFrame(data)
        # Work out total F
        df['F_vec'] = np.linalg.norm([df['%s_vec' % d] for d in self.dimensions], axis=0)

        if combined:
            return df.groupby(['x','y','z']).sum().reset_index()
        else:
            return df

    def combine_masses(self, mrs=None):
        """ Combines the mass components into a single mass, to speed up analysis of large spaces """
        if mrs is None and fit_ratios:
            mrs = self.fit_ratios
        else:
            mrs = dict([(l, 1) for l in self.mass_labels])

        self.mass_components = np.array([np.sum([c*mrs[self.mass_labels[i]] for i, c in enumerate(self.mass_components)], axis=0),])

        self.mass_components.flags.writeable = False
        self.mass_labels = ['combined',]
        self.results = {}

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

    @property
    def mass_sum(self):
        return np.sum(self.mass_components, axis=0)

    def mass_profile(self, stack=None, cmap='Spectral_r', ax=None):
        """ Heatmap of mass over space """
        if stack is None: stack = self.space.center[0]
        sns.heatmap(self.mass_sum[stack],
            cmap=cmap, ax=ax,
            square=True, norm=LogNorm(),
            xticklabels=None, yticklabels=None)
            #xticklabels=self.space.x, yticklabels=self.space.coords[1]*self.space.scale)

    def save(self, filename, masses=True):
        import pickle
        if masses:
            self.log("Saving %s masses" % filename)
            with open("%s.npy" % filename, 'wb') as f:
                np.save(f, self.mass_components)
        else:
            self.log("Throwing masses away!")
        self.mass_components = None

        self.log("Saving %s pickle" % filename)
        with open('%s.pickle' % filename, 'wb') as fh:
            pickle.dump(self, fh)
        
        self.log("Complete")






