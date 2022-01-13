import saver
import numpy as np
from models.milky import MilkyWay

if __name__ == '__main__':
    total_points = 300
    total_radius = 40
    calc_radius = 25
    percent = calc_radius/total_radius

    sim = MilkyWay(points=total_points, radius=total_radius, cp=print)
    rl = sim.space.radius_list

    sim.analyse(rl[int(len(rl)*(1-percent)):], min_mass=100)
    saver.save('milky_slice', sim)