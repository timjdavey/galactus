import sys
sys.path.append("../")

from models.galaxy import Galaxy
from references.milkyway import profiles

if __name__ == '__main__':
    mass_points = 100
    calc_points = 15
    
    mass_radius = 50
    calc_radius = 50
    
    minimum_mass = 0
    filename = 'milky'

    pname = 'mcmillian2011best'
    
    sim = Galaxy(profiles=profiles[pname], points=mass_points, radius=mass_radius, cp=print, combine_masses=False)
    sim.analyse(sim.radius_points(calc_radius, calc_points), minimum_mass)
    sim.analyse(sim.radius_points(calc_radius, calc_points), minimum_mass, alt=True)

    #sim.save("%s_%s_%s_%s" % (filename, pname, component, mass_points))