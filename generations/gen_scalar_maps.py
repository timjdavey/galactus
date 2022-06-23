import sys
sys.path.append("../")

from models.sparc.profile import quality_profiles
from models.memory import memory_usage


if __name__ == '__main__':
    qp = quality_profiles(2)
    count = len(qp)
    points = 300
    ycut = 1

    for i, p in enumerate(qp.values()):
        try:
            print('%s of %s loading %s %s' % (i, count, p.uid, memory_usage()))
            scalar = p.generate_scalar_map(points, ycut, load=True, DIR='')
        except FileNotFoundError:
            print('%s of %s, %s not found, generating instead' % (i, count, p.uid))
            p.generate_scalar_map(points, ycut, save=True, DIR='')
