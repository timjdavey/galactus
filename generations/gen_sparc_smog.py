import sys
sys.path.append("../")

import time
from models.sparc.profile import quality_profiles
from models.sparc.galaxy import save_smog
from models.memory import memory_usage
from models.load import load_sparc

if __name__ == '__main__':
    qp = quality_profiles(2)
    count = len(qp)
    points = 300
    z = 1
    DIR = ''

    for i, p in enumerate(qp.values()):
        print('%s of %s. %s' % (i, count, p.uid))
        namespace = "%ssparc_smog_%s_%s" % (DIR, points, z)
        try:
            load_sparc(namespace, profiles={p.uid: p}, directory=DIR, ignore=False)
            print('file found, skipping to next')
        except FileNotFoundError:
            tic = time.perf_counter()
            save_smog(p, points, z, "%s_%s" % (namespace, p.uid))
            toc = time.perf_counter()
            print('completed in %s %s' % (toc-tic, memory_usage()))
