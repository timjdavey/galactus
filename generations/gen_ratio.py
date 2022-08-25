import sys
sys.path.append("../")

import time
from models.params import points, z
from models.load import load_sparc
from models.sparc.profile import quality_profiles
from models.sparc.galaxy import generate_pmog
from models.workers import ratio_worker


if __name__ == '__main__':
    profiles = quality_profiles(3)
    filename = 'ratio/%s_%s_%s'
    count = len(profiles)
    errors = []

    print("Starting ratio generation")
    try:
        fits = load_sparc('baseline/%s_%s' % (points, z), ignore=False)
    except FileNotFoundError:
        raise FileNotFoundError("Please run gen_baseline.py first")
    for i, name in enumerate(profiles.keys()):
        try:
            tic = time.time()
            profile = profiles[name]
            gal = generate_pmog(profile, points, z, ratio_worker, fit_ratios=fits[name].fit_ratios)
            gal.save(filename % (points, z, name), masses=False)
            toc = time.time()
            print("%s of %s %s %.1fs" % (i, count, filename % (points, z, name), toc-tic))
        except IndexError:
            errors.append(name)

    print("Finished, with errors %s" % errors)
