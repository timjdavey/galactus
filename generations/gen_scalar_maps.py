import sys
sys.path.append("../")

from models.sparc.profile import quality_profiles
from models.memory import memory_usage


if __name__ == '__main__':
    qp = quality_profiles(2)
    count = len(qp)
    points = 300
    ycut = 2 # cut this to 2 and it's dramatically faster
    force = True

    for i, p in enumerate(qp.values()):
        try:
            if force: raise FileNotFoundError
            print('%s of %s trying to load %s %s' % (i, count, p.uid, memory_usage()))
            p.generate_maps(points, ycut, load=True, DIR='', cp=print)
        except FileNotFoundError:
            print('%s of %s, generating %s' % (i, count, p.uid))
            p.generate_maps(points, ycut, save=True, DIR='', cp=print)
