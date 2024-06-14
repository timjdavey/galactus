import copy
import time
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
from .workers import newtonian_worker, initializer

Gkg = 6.67430 * (10**-11)  # m3 kg-1 s-2
Gsolar = 4.30091 * (10**-6)  # kpc Ms-1 (km/s)2


class Simulation:
    """
    Main simulation object
    """

    dimensions = ("z", "y", "x")

    def __init__(
        self,
        masses,
        space,
        smaps=None,
        worker=newtonian_worker,
        mass_labels=None,
        G=Gsolar,
        cp=None,
        ignore_odd=False,
    ):
        if space.is_even and not ignore_odd:
            raise ValueError(
                "Space %s is not odd. Because of symmetries, please make all spacial dimensions odd or set `ignore_odd` to True."
                % str(space.points)
            )

        if isinstance(masses, list):
            masses = np.array(masses)
        masses.flags.writeable = False  # makes masses a constant
        self.mass_components = masses
        self.mass_sums = [np.sum(m) for m in self.mass_components]
        if mass_labels is not None:
            self.mass_labels = mass_labels
        else:
            self.mass_labels = ["mass %s" % i for i in range(len(masses))]
        self.space = space
        self.smaps = smaps
        self.worker = worker
        self.G = G
        self.cp = cp
        self.results = {}
        self.fit_ratios = {"bul": 1, "disk": 1, "gas": 1}

    def analyse(self, sub_list=None, verbose=False, processes=8):
        """Main function to generate the results"""
        tic = time.perf_counter()
        point_list = self.space.list if sub_list is None else sub_list
        tasks = self.space.count if sub_list is None else len(sub_list)
        count = 0

        # if doing the whole space, use chunksize, otherwise keep it lean and mean
        chunksize = tasks // (processes**2) + 1

        with Pool(
            processes, initializer, (self.mass_components, self.space.scale, self.smaps)
        ) as pool:
            for p, r in pool.imap_unordered(self.worker, point_list, chunksize):
                if verbose:
                    self.log("%s of %s completed %s" % (count, tasks, memory_usage()))
                self.results[p] = r
                count += 1

        toc = time.perf_counter()
        self.log("completed in %.2f seconds" % (toc - tic))

    def space_map(self, result_index=0, mass_component=0, fast=True):
        """Generates an array of the scalar values for all space"""
        smap = self.space.blank()
        fast = self.space.count != len(self.results)

        for p, vals in self.results.items():
            v = vals[mass_component][result_index]
            if fast:
                z, y, x = p
                for i, j in ((y, x), (x, y)):
                    for k in (z, -z - 1):
                        smap[k][i][j] = v
                        smap[k][-i - 1][j] = v
                        smap[k][i][-j - 1] = v
                        smap[k][-i - 1][-j - 1] = v
            else:
                smap[p] = v
        return smap * self.G

    def space_maps(self, mass_index=0):
        """Generates for all maps, for a given mass_component (default just 1 as likely combined)"""
        return [
            self.space_map(i)
            for i in range(len(list(self.results.values())[0][mass_index]))
        ]

    def dataframe(self, mass_ratios=False, combined=False):
        """Returns the results as a dataframe"""
        if len(self.results) == 0:
            raise ValueError("No results yet, please run .analyse()")
        # So can override mass_ratios
        # both applied later on dataframe creation
        if mass_ratios is False:
            mass_ratios = dict([(c, 1) for c in self.mass_labels])

        data = []

        for ijk, result in self.results.items():
            rr = {}
            for mi, component_result in enumerate(result):
                component = self.mass_labels[mi]
                rr = dict([(d, ijk[di]) for di, d in enumerate(self.dimensions)])
                rr["component"] = component

                # for each of the masses
                for dim, value in enumerate(component_result[0]):
                    rr["%s_vec" % self.dimensions[dim]] = (
                        value
                        * self.G
                        * mass_ratios[component]
                        # we treat these seperately
                        # as they are outside of Lelli's calculations
                        * self.fit_ratios[component]
                    )
                data.append(rr)

        df = pd.DataFrame(data)
        # Work out total F
        df["F_vec"] = np.linalg.norm(
            [df["%s_vec" % d] for d in self.dimensions], axis=0
        )

        if combined:
            return df.groupby(["x", "y", "z"]).sum().reset_index()
        else:
            return df

    def log(self, *args, **kwargs):
        """Outputs to notebook"""
        if self.cp is not None:
            self.cp(*args, **kwargs)

    def save(self, folder, masses=False):
        import pickle

        filename = "%s/%s_%s" % (folder, self.space.points, self.space.z)

        if masses:
            self.log("Saving %s masses" % filename)
            with open("%s.npy" % filename, "wb") as f:
                np.save(f, self.mass_components)
        else:
            self.log("Throwing masses away!")
            self.mass_components = None

        self.log("Saving %s pickle" % filename)
        with open("%s.pickle" % filename, "wb") as fh:
            pickle.dump(self, fh)

        self.log("Complete")
