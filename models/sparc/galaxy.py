import pickle
import numpy as np
import pandas as pd

from models.space import Space
from models.galaxy import Galaxy
from models.load import load_sparc
from models.sparc.profile import MASS_RATIOS

def generate_galaxy(profile, space_points=300, z=1, excess_ratio=1.5, calc_points=0,
        rotmass_points=True, cp=None):
    """
    Generates a sparc galaxy given a profile
    """
    uid = profile.uid
    space = Space((z,space_points,space_points), profile.max_r*2*excess_ratio/space_points)
    masses, labels = profile.masses(space)
    
    sim = Galaxy(masses, space, mass_labels=labels, cp=cp)
    sim.profile = profile
    sim.name = uid
    
    if rotmass_points:
        sim.analyse(profile.rotmass_points(space))

    if calc_points:
        sim.analyse(sim.radius_points(profile.max_r*excess_ratio, calc_points))
    
    if z > 1:
        sim.fit_ratios = profile.fit_simulation(sim)

    return sim


def galaxy_scalar_map(profile, space_points=300, z=1, excess_ratio=1.5, cp=None, load=True, save=True, DIR='generations/'):
    """
    Creates a scalar map galaxy
    """
    
    namespace = [str(a) for a in ('sparc_map', space_points, z, profile.uid)]

    if load:
        return load_sparc(namespace, profiles={profile.uid: profile}, ignore=False)[profile.uid]
    else:
        sim = generate_galaxy(profile, space_points, z, excess_ratio,
                rotmass_points=z>1, calc_points=0, cp=cp) # do no analysis
    
        if z > 1:
            # if not flat
            # need to fit simulation to Lelli mass model components
            sim.combine_masses(sim.fit_ratios)
        else:
            # otherwise combine masses using 0.5, 0.7, 1.0 ratios
            sim.combine_masses(MASS_RATIOS)
        
        sim.analyse(sim.space.slice_list, verbose=False)

        if save:
            # can rebuild masses if needed
            # from profile * fit_ratios
            sim.save('%s%s' % (DIR, namespace), masses=False)
        return sim




