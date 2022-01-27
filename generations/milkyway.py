import sys
sys.path.append("../")

from models.galaxy import Galaxy
from references.milkyway import profiles

if __name__ == '__main__':
    pname = 'huang2016'

    mass_points = 800
    mass_radius = 50

    point_sections = [
        # points, upto radius
        (15, 5), 
        (20, 25),
        (10, mass_radius),
    ]
    
    sim = Galaxy(profiles=profiles[pname], points=mass_points, radius=mass_radius, cp=print)
    
    for calc_points, calc_radius in point_sections:
        sim.analyse(sim.radius_points(calc_radius, calc_points))
    
    filename = "_".join([str(s) for s in [pname, mass_points, mass_radius]])
    sim.save(filename)