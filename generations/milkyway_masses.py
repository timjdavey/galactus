import sys
sys.path.append("../")

import numpy as np
import copy
from models.galaxy import Galaxy
from references.milkyway import profiles

if __name__ == '__main__':
    pname = 'mcmillian2011best'

    mass_points = 400
    for boost in np.linspace(0.5, 5, 5):

        mass_radius = 40*boost
    
        point_sections = [
            # points, upto radius
            #(10, 25),
            (10, mass_radius),
        ]
    
        profile = copy.deepcopy(profiles[pname])
        profile['buldge']['params']['rcut'] *= boost
        profile['thick']['params']['Rd'] *= boost
        profile['thin']['params']['Rd'] *= boost
        sim = Galaxy(profiles=profile, points=mass_points, radius=mass_radius, cp=print)
    
        for calc_points, calc_radius in point_sections:
            sim.analyse(sim.radius_points(calc_radius, calc_points))
    
        sim.save("massestest_%s" % boost)