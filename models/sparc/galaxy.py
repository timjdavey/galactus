import pickle
import numpy as np
import pandas as pd

from models.space import Space
from models.galaxy import Galaxy


def generate_galaxy(profile, space_points=300, calc_points=0, rot_fit=True,
        rotmass_points=True, flat=True, zcut=10, ycut=1, excess_ratio=1.2, cp=None):
    """
    Generates a sparc galaxy given a profile
    """
    uid = profile.uid
    z = 1 if flat else space_points//zcut
    space = Space((z,space_points//ycut,space_points), profile.max_r*2*excess_ratio/space_points)
    masses, labels = profile.masses(space)
    
    sim = Galaxy(masses, space, mass_labels=labels, cp=cp)
    sim.profile = profile
    sim.name = uid
    
    if rotmass_points:
        sim.analyse(profile.rotmass_points(space))

    if calc_points:
        sim.analyse(sim.radius_points(profile.max_r*excess_ratio, calc_points))
    
    if rot_fit:
        sim.fit_ratios = profile.fit_simulation(sim)

    return sim
