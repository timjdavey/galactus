import pickle
import numpy as np
import pandas as pd

from models.space import Space
from models.galaxy import Galaxy
from models.load import load_sparc
from models.sparc.profile import MASS_RATIOS
from models.workers import map_worker, newtonian_worker

def generate_galaxy(profile, space_points=300, z=1, excess_ratio=1.5, calc_points=0,
        rotmass_points=True, cp=None, worker=newtonian_worker, fit=False):
    """
    Generates a sparc galaxy given a profile

    :rotmass_points: bool, whether to run for the points in the sparc rotmass file
    :calc_points: int, total number of even points to create if want a general picture of profile
    :cp: for logging
    :fit: whether to fit the simulation to Lelli model for non-flat models
    """
    uid = profile.uid

    space = Space((z,space_points,space_points), profile.max_r*2*excess_ratio/space_points)
    masses, labels = profile.masses(space)
    
    sim = Galaxy(masses, space, worker=worker, mass_labels=labels, cp=cp)
    sim.profile = profile
    sim.name = uid
    
    if rotmass_points:
        sim.analyse(profile.rotmass_points(space))

    if calc_points:
        sim.analyse(sim.radius_points(profile.max_r*excess_ratio, calc_points))
    
    if fit and z > 1:
        sim.fit_ratios = profile.fit_simulation(sim)

    return sim


def generate_map(profile, space_points=300, z=1,
    excess_ratio=1.5, fit_ratios=None, cp=None, fast=True):
    """
    Creates a map galaxy
    """
    # do not analyse, as need to combine masses first
    sim = generate_galaxy(profile, space_points, z, excess_ratio,
        rotmass_points=False, calc_points=0, cp=cp, worker=map_worker, fit=False)
    
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
    smap = generate_map(profile, space_points, z, fit_ratios=fit_ratios)
    vals = smap.space_maps()
    vals.append(pmog_k)
    gal = Galaxy(smap.mass_components, smap.space, worker=worker, mass_labels=smap.mass_labels, smaps=vals)
    gal.analyse(smap.profile.rotmass_points(smap.space))
    gal.profile = profile
    return gal

