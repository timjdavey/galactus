import saver
from models.milky import MilkyWay

if __name__ == '__main__':
    for points in (11,21):
        sim = MilkyWay(points=points, radius=points, cp=print)
        sim.analyse(sim.space.radius_list, multi=True)
        saver.save('milky_scale_%s' % points, sim)