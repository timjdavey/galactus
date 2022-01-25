import sys
sys.path.append("../")

from models.galaxy import Galaxy
from references.milkyway import profiles

if __name__ == '__main__':
    mass_points = 50
    calc_points = 15
    
    mass_radius = 20
    calc_radius = 20
    
    minimum_mass = 0
    name = 'milkytest'
    
    for pname in ('mcmillian2011best',):
        pfiles = profiles[pname]
        print("Starting %s" % pname)
        sim = Galaxy(profiles=pfiles, points=mass_points, radius=mass_radius, cp=print)
        sim.analyse(sim.radius_points(calc_radius, calc_points), minimum_mass, alt=True)
        sim.analyse(sim.radius_points(calc_radius, calc_points), minimum_mass)
        sim.save("%s_%s_%s" % (name, pname, mass_points))