import sys
sys.path.append("../")

from models.galaxy import Galaxy
from references.milkyway import profiles

if __name__ == '__main__':
    mass_points = 600
    calc_points = 20
    
    mass_radius = 50
    calc_radius = 25
    
    minimum_mass = 0.00001
    name = 'milky'
    
    for pname in ('mcmillian2011best','huang2016'):
        pfiles = profiles[pname]
        print("Starting %s" % pname)
        sim = Galaxy(profiles=pfiles, points=mass_points, radius=mass_radius, cp=print)
        sim.analyse(sim.radius_points(calc_radius, calc_points), minimum_mass)
        sim.save("%s_%s_%s" % (name, pname, mass_points))