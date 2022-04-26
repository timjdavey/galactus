from models.space import Space
from models.galaxy import Galaxy


def generate_galaxy(profile, space_points=5000, calc_points=20, rotmass_points=False, cp=None):
    """
    Generates a galaxy given a profile
    """
    uid = profile.uid
    space = Space((1,space_points,space_points), profile.max_r*4/space_points)
    masses, labels = profile.masses(space)
    
    sim = Galaxy(masses, space, mass_labels=labels, cp=cp)
    sim.profile = profile
    sim.name = uid
    
    if calc_points:
        sim.analyse(sim.radius_points(profile.max_r*1.5, calc_points))
    if rotmass_points:
        sim.analyse(profile.rotmass_points(space))
    
    return sim