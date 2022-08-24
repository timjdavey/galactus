import pickle
import numpy as np
import pandas as pd

from models.space import Space
from models.galaxy import Galaxy
from models.load import load_sparc
from models.sparc.profile import MASS_RATIOS
from models.workers import map_worker

def generate_galaxy(profile, space_points=300, z=1, excess_ratio=1.5, calc_points=0,
        rotmass_points=True, cp=None, worker=None):
    """
    Generates a sparc galaxy given a profile

    :rotmass_points: bool, whether to run for the points in the sparc rotmass file
    :calc_points: int, total number of even points to create if want a general picture of profile
    :cp: for logging
    """
    uid = profile.uid

    space = Space((z,space_points,space_points), profile.max_r*2*excess_ratio/space_points)
    masses, labels = profile.masses(space)
    
    sim = Galaxy(masses, space, mass_labels=labels, cp=cp)
    if worker: sim.worker = worker
    sim.profile = profile
    sim.name = uid
    
    if rotmass_points:
        sim.analyse(profile.rotmass_points(space))

    if calc_points:
        sim.analyse(sim.radius_points(profile.max_r*excess_ratio, calc_points))
    
    if z > 1:
        sim.fit_ratios = profile.fit_simulation(sim)

    return sim


def generate_map(profile, space_points=300, z=1,
    excess_ratio=1.5, fit_ratios=None, cp=None, fast=True):
    """
    Creates a map galaxy
    """
    sim = generate_galaxy(profile, space_points, z, excess_ratio,
        # do no analysis
        rotmass_points=False, calc_points=0, cp=cp, worker=map_worker)
    
    if z > 1:
        # if not flat
        # need to fit simulation to Lelli mass model components
        sim.combine_masses(fit_ratios)
    else:
        # otherwise combine masses using 0.5, 0.7, 1.0 ratios
        sim.combine_masses(MASS_RATIOS)
    
    if fast:
        # do for slice & infer in galaxy_smog
        sim.analyse(sim.space.symmetric_points)
    else:
        # otherwise calculate for all points
        sim.analyse()
    return sim


def generate_pmog(profile, space_points, z, worker, pmog_k=25000, fit_ratios=None):
    """ Generates for a pmog galaxy """
    smap = galaxy_scalar_map(profile, space_points, z, fit_ratios=fit_ratios)
    vals = smap.space_maps
    vals.append(pmog_k)
    gal = Galaxy(smap.mass_components, smap.space, mass_labels=smap.mass_labels, smaps=vals)
    gal.analyse(smap.profile.rotmass_points(smap.space))
    gal.profile = profile
    return gal

