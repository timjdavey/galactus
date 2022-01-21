import sys
sys.path.append("../")

from models.galaxy import Galaxy
from references.milkyway import profiles

if __name__ == '__main__':
    total_points = 300
    total_radius = 40
    calc_radius = 25
    minimum_mass = 0.00001

    name = 'milky_'
    percent = calc_radius/total_radius

    for pname, pfiles in [('huang2016', profiles['huang2016']),]:#profiles.items():
        print("Starting %s" % pname)
        sim = Galaxy(profiles=pfiles, points=total_points, radius=total_radius, cp=print)
        rl = sim.space.radius_list
        sim.analyse(rl[int(len(rl)*(1-percent)):], min_mass=minimum_mass)
        sim.rotate('x','z')
        sim.name = pname
        sim.save("%s%s100" % (name, pname))