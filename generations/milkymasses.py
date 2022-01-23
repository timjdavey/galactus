import sys
sys.path.append("../")

import numpy as np
from models.galaxy import Galaxy
from references.milkyway import profiles

if __name__ == '__main__':
    pname = 'huang2016'

    mass_points = 100
    calc_points = 10
    
    mass_radius = 50
    calc_radius = 25
    
    minimum_mass = 0.00001
    name = 'milkymass'
    percent = calc_radius/mass_radius
    
    sim = Galaxy(profiles=profiles[pname], points=mass_points, radius=mass_radius, cp=print)
    sim.analyse(sim.space.radius_list[:20])
    sim.save("%s_%s_%s" % (name, pname, mass_points))