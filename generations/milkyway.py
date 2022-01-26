import sys
sys.path.append("../")

from models.galaxy import Galaxy
from references.milkyway import profiles

if __name__ == '__main__':
    mass_points = 800
    calc_points = 24
    processes = 4
    
    mass_radius = 50
    calc_radius = 50
    
    pname = 'mcmillian2011best'
    
    sim = Galaxy(profiles=profiles[pname], points=mass_points, radius=mass_radius, cp=print, combine_masses=False)
    sim.analyse(sim.radius_points(calc_radius, calc_points), processes=processes)
    filename = "_".join([str(s) for s in [pname, mass_points, calc_points, mass_radius, calc_radius]])
    sim.save(filename)