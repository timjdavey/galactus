import sys
sys.path.append("../")

from models.galaxy import Galaxy
from references.milkyway import profiles

if __name__ == '__main__':
    for mass_points in (500,):

        calc_points = 10
    
        mass_radius = 50
        calc_radius = 25
    
        minimum_mass = 0.00001
        pname = 'huang2016'
        pfiles = profiles[pname]
    
        name = 'milkytest'
        percent = calc_radius/mass_radius
    
        print("Starting %s" % pname)
        sim = Galaxy(profiles=pfiles, points=mass_points, radius=mass_radius, cp=print)
        rl = sim.space.radius_list
        max_point = len(rl)*percent
        sim.analyse(rl[:int(max_point)+1:max(int(max_point/calc_points),1)], min_mass=minimum_mass)
        sim.name = pname
        sim.save("%s_%s_%s" % (name, pname, mass_points))