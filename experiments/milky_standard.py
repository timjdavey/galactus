import saver
from models.milky import MilkyWay

if __name__ == '__main__':
    sim = MilkyWay(points=201, radius=50, cp=print)
    sim.analyse(sim.space.radius_list, multi=True)
    saver.save('milky_standard', sim)