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
    points = 401
    zs = (1,11)
    DIR = ''
    force = False

    for z in zs:
        for i, p in enumerate(qp.values()):
            print('%s of %s. %s' % (i, count, p.uid))
            namespace = "%ssparc_smog_%s_%s" % (DIR, points, z)
            try:
                if force: raise FileNotFoundError
                load_sparc(namespace, profiles={p.uid: p}, directory=DIR, ignore=False)
                print('%s found, skipping to next' % namespace)
            except FileNotFoundError:
                tic = time.perf_counter()
                save_smog(p, points, z, "%s_%s" % (namespace, p.uid))
                toc = time.perf_counter()
                print('completed %s in %.1fs using %s' % (namespace, toc-tic, memory_usage()))
