import sys
sys.path.append("../")

from models.sparc.profile import quality_profiles
from models.sparc.galaxy import galaxy_scalar_map
from models.memory import memory_usage


if __name__ == '__main__':
    qp = quality_profiles(2)
    count = len(qp)
    points = 300
    z = 1
    DIR = ''

    for i, p in enumerate(qp.values()):
        try:
            print('%s of %s trying to load %s %s' % (i, count, p.uid, memory_usage()))
            galaxy_scalar_map(p, points, z, DIR=DIR, load=True)
        except FileNotFoundError:
            print('%s of %s, generating %s' % (i, count, p.uid))
            galaxy_scalar_map(p, points, z, DIR=DIR, save=True, cp=print)
