from models.space import Space
from models.galaxy import Galaxy

def generate_galaxy(profile, space_points=500, calc_points=20, radius=1, zcut=10, fit=True, cp=None):
    space = Space((space_points//zcut, space_points, space_points), radius*2/space_points)
    masses, mass_labels = profile.masses(space)
    simulation = Galaxy(masses, space, cp=cp, mass_labels=mass_labels)
    
    if calc_points:
        simulation.analyse(simulation.radius_points(radius, calc_points))
    
    if fit:
        profile.fit_simulation(simulation)
    return simulation