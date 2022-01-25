import sys
sys.path.append("../")

from models.galaxy import Galaxy
from references.milkyway import profiles

if __name__ == '__main__':
    mass_points = 200
    calc_points = 20
    
    mass_radius = 100
    calc_radius = 100
    
    minimum_mass = 0
    filename = 'milky'

    pname = 'mcmillian2011best'
    
    sim = Galaxy(profiles=profiles[pname], points=mass_points, radius=mass_radius, cp=print, combine_masses=False)
    sim.analyse(sim.radius_points(calc_radius, calc_points), minimum_mass)
    #sim.save("%s_%s_%s_%s" % (filename, pname, component, mass_points))