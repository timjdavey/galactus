import saver
from models.milky import MilkyWay

if __name__ == '__main__':
    total_points = 201
    total_radius = 50
    calc_radius = 30

    sim = MilkyWay(points=total_points, radius=total_radius, cp=print)
    calc_points = len(sim.space.radius_list)*(1-total_radius/calc_radius)
    sim.analyse(sim.space.radius_list[int(calc_points):], multi=True)
    saver.save('milky_center', sim)