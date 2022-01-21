import numpy as np
import pandas as pd
import seaborn as sns
import copy
import matplotlib.pyplot as plt
from functools import cached_property, partial
from matplotlib.colors import LogNorm
from .pool import pool
from .space import Space


def gravity_worker(iop, total, nonzero_masses, mass_components, space_scale, cp):
    """
    Multiprocessing function for calculating gravity
    for all the non_zero masses within mass_sum
    and for a specific `observed_points` in space
    """
    i, observed_points = iop
    observed_distance = np.array(observed_points)*space_scale

    if cp is not None: cp("%s of %s, %.1f%%" % (i, total, i*100/total))

    # save results by mass component
    results = {}
    for mi in range(len(mass_components)):
        rr = {'F': 0}
        for di in range(len(observed_points)):
            rr[di] = 0
            rr['a%s' % di] = 0
        results[mi] = rr

    # calculate for all nonzero_mass points
    for mass_points in nonzero_masses:
        
        if np.array_equal(mass_points,observed_points):
        # the mass of a certain grid point
        # has no impact on it's own grid point
        # to avoid infinities etc
        # so leave as 0
            pass
        else:
            mass_distance = np.array(mass_points)*space_scale

            deltas = observed_distance-mass_distance
        
            r2 = np.sum([d**2 for d in deltas])
            r = r2**0.5

            for mi, mass_comps in enumerate(mass_components):
                M = mass_comps[tuple(mass_points)]
                F = M/r2
            
                ijk = tuple(observed_points)
                for di, delta in enumerate(deltas):
                    g = -F*delta/r
                    results[mi][di] += g
                    results[mi]['a%s' % di] += np.abs(g)
            
                results[mi]['F'] += np.abs(F)

    return (ijk, results)



class Simulation:
    """
    Main simulation object
    """
    def __init__(self, masses, space, mass_labels=None, G=1, cp=None):
        masses = np.array(masses)
        if masses.shape == tuple(space.points):
            masses = [masses,]
        else:
            for i, mass in enumerate(masses):
                if mass.shape != space.points:
                    raise ValueError("mass %s shape does not match space grid (%s, %s)" % (i, mass.shape, space.points))

        self.mass_components = masses
        self.space = space
        self.G = G
        self.cp = cp
        self.mass_labels = mass_labels

    def analyse(self, sub_list=None, min_mass=0, shortcut=False, multi=True):
        """
        Does the main bulk of the analysis

        :sub_list: array_like of points to calculate for, rather than whole of space
        :min_mass: 0, only calculate for points with mass greater than this value
        :shortcut: to use with radius_results, where you don't need full 3d makeup

        first assumption is all masses
        in each grid point
        are summed to a single mass for that grid
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

        space = self.space
        
        if sub_list is not None:
            point_list = sub_list
        else:
            point_list = space.list
    
        mass_sum = self.mass_sum()
        mass_list = np.transpose(np.where(mass_sum > min_mass))
        
        worker = partial(gravity_worker,
            total=len(point_list),
            nonzero_masses=mass_list,
            mass_components=self.mass_components,
            space_scale=space.scale,
            cp=self.cp)

        result_list = pool(worker, enumerate(point_list), multi)

        self.result_list = result_list
        self.mass_list = mass_list
        self.sub_list = sub_list

        self.__fields = None
        self.__sums = None
        self.__dataframe = None

    @property
    def fields(self):
        if self.__fields is None:
            space = self.space
            fields = dict([(d, space.blank()) for d in space.dimensions])
            for d in self.space.dimensions:
                fields['a%s' % d] = space.blank()
            fields['F'] = space.blank()
        
            # fields by mass component
            mass_fields = {}
            for mi in range(len(self.mass_components)):
                mass_fields[mi] = copy.deepcopy(fields)
        
            self.log('combine point results')
            for ijk, mass_results in self.result_list:
                for mi, dimension_results in mass_results.items():
                    for k,v in dimension_results.items():
                        mass_fields[mi][k][tuple(ijk)] += v
            self.__fields = mass_fields
        
            self.log('sum results')
            sums = {'mass': self.mass_sum()}
            for di in dimension_results.keys():
                all_dimension = [mass_fields[mi][di] for mi in range(len(self.mass_components))]
                sums[di] = np.sum(all_dimension, axis=0)
            
            self.__sums = sums
        return self.__fields

    @property
    def sums(self):
        if self.__sums is None:
            self.fields
        return self.__sums

    def dataframe(self):
        results = []
        for ijk, result in self.result_list:
            sums = {
                'z': ijk[0],
                'y': ijk[1],
                'x': ijk[2],
                'ijk': ijk,
                'component': 'sum',
            }
            for mass_index, mass_res in result.items():
                rr = sums.copy()
                if self.mass_labels is not None:
                    rr['component'] = self.mass_labels[mass_index] 
                else:
                    rr['component'] = mass_index
                for k,v in mass_res.items():
                    rr[k] = v
                    if mass_index == 0:
                        sums[k] = v
                    else:
                        sums[k] += v
                results.append(rr)
            results.append(sums)
        return pd.DataFrame(results)

    def mass_sum(self):
        return np.sum(self.mass_components, axis=0)

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
        self.log('sums')
        dic = self.sums
        self._sums = copy.deepcopy(dic)
        for key, value in dic.items():
            dic[key] = np.rot90(dic[key], axes=(d1,d2))
        self.__sums = dic


        # space
        self.log('space')
        self._space = self.space
        old_points = self.space.points
        new_points = list(old_points)
        new_points[d2] = old_points[d1]
        new_points[d1] = old_points[d2]
        self.space = Space(new_points, self.space.scale)


        # mass
        self.log('mass components')
        self._fields = copy.deepcopy(self.fields)
        for mi, dic in self.fields.items():
            for key, value in dic.items():
                dic[key] = np.rot90(dic[key], axes=(d1,d2))
            self.fields[mi] = dic
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
        cmap = 'icefire' if vmin < 0 else 'cubehelix'

        for i, grid in enumerate(grids):
            f = sns.heatmap(grid, ax=axes[i][col], vmin=vmin, vmax=vmax,
                cbar= (i == 0), cmap=cmap)
            if i == 0:
                f.set(title=title)
        self.log()

    def heatmaps(self, slices=None, image_scale=15, show=(0,1,2,'mass')):
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
            if k in show or show is None:
                if slices is None:
                    grids = self.sums[k] 
                else:
                    grids = self.sums[k][slices[0]:slices[1]:slices[2]]
    
                self._heatmap(grids, i, axes, title=k)

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

    def save(self, filename):
        import pickle
        print("Saving %s" % filename)
        with open('%s.pickle' % filename, 'wb') as fh:
            pickle.dump(self, fh)

        print("Complete")






