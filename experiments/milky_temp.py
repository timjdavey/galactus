import saver
from models.milky import MilkyWay

if __name__ == '__main__':
    total_points = 101
    total_radius = 30
    calc_radius = 12
    filename = 'milky_temp'

    sim = MilkyWay(points=total_points, radius=total_radius, cp=print)
    calc_points = len(sim.space.radius_list)*(1-calc_radius/total_radius)
    sim.analyse(sim.space.radius_list[int(calc_points):], multi=True)
    saver.save(filename, sim)